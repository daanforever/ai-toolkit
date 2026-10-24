"""
Probe: does the step-0 (baseline) sample of zimage_diffsynth depend on the dataset?

At step 0 lora_up (B) is all-zero, so the baseline sample must equal the base model output and
must not depend on dataset content. This script checks that with the real trainer path:

  A         dataset = temp/test_train pairs except the last one, caches cleared
  B         dataset = A + the last pair, caches cleared
  A_repeat  same as A, caches cleared (control; skip with --no-control)

Each run is a fresh subprocess: ``run_job`` builds the dataloader (latent + text-embedding caching
to disk, ``unload_text_encoder``, sample-prompt cache), then the trainer requests the baseline
sample. A probe on ``BaseSDTrainProcess.sample`` records the sample inputs, lets the baseline sample
render, and stops the job before any optimizer step.

Recorded per run (``result.json``) and compared at the end:
  - lora_up (B) max |w| for the main and the sampling network (must be 0)
  - md5 of the conditional / unconditional sample-prompt embeddings
  - byte-checksum of the VAE and of the DiT that renders the sample
  - the step-0 image (``step0.png``): bit-identical or max / mean pixel difference

Run (repo root):
  venv\\Scripts\\python.exe -m testing.zimage_step0_dataset_probe

Env: ZIMAGE_DIFFSYNTH_MODEL_PATH, ZIMAGE_DIFFSYNTH_SAMPLING_PATH (same as testing/ real-model tests).
Requires CUDA, local Z-Image weights and ``temp/test_train`` (png + txt pairs).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.test_flowmatch_snr_real_model import (  # noqa: E402
    _resolve_model_paths,
    _user_flowmatch_job_config,
)

SOURCE_DATASET_DIR = REPO_ROOT / "temp" / "test_train"
PROBE_ROOT = REPO_ROOT / "temp" / "step0_dataset_probe"
CACHE_DIR_NAMES = ("_latent_cache", "_t_e_cache")
JOB_NAME = "zimage_step0_dataset_probe"
SAMPLE_PROMPT = "dog"
SAMPLE_SEED = 42
RESULT_NAME = "result.json"
IMAGE_NAME = "step0.png"
_CHECKSUM_CHUNK = 1 << 26


class BaselineSampleDone(Exception):
    """Raised by the probe right after the baseline sample, before any training step."""


def _log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------- fingerprints (worker)


def _tensor_byte_sum(tensor: torch.Tensor) -> int:
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    total = 0
    for start in range(0, raw.numel(), _CHECKSUM_CHUNK):
        total += int(raw[start:start + _CHECKSUM_CHUNK].sum(dtype=torch.int64))
    return total


def _raw_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Tensor subclasses (torchao / quanto quantized weights) expose their payload tensors."""
    if hasattr(tensor, "__tensor_flatten__"):
        inner_names, _ = tensor.__tensor_flatten__()
        parts: List[torch.Tensor] = []
        for inner_name in inner_names:
            parts.extend(_raw_tensors(getattr(tensor, inner_name)))
        return parts
    return [tensor]


def _module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.md5()
    named = list(module.named_parameters()) + list(module.named_buffers())
    for name, tensor in named:
        for part in _raw_tensors(tensor):
            digest.update(
                f"{name}|{tuple(part.shape)}|{part.dtype}|{_tensor_byte_sum(part)}".encode()
            )
    return digest.hexdigest()


def _tensors_digest(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return hashlib.md5("|".join(_tensors_digest(v) for v in value).encode()).hexdigest()
    cpu = value.detach().cpu().contiguous().reshape(-1)
    return hashlib.md5(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _lora_up_max_abs(network) -> float:
    ups = [m.lora_up.weight for m in network.unet_loras if hasattr(m, "lora_up")]
    if not ups:
        raise RuntimeError("network has no lora_up modules")
    return max(float(w.detach().abs().max()) for w in ups)


def _prompt_embed_digests(sd) -> Dict[str, str]:
    from toolkit.sample_prompts_cache import load_sample_prompt_pair

    if not sd.sample_prompts_cache:
        raise RuntimeError("sample_prompts_cache is empty: expected cached sample prompt embeddings")
    cond, uncond = load_sample_prompt_pair(sd.sample_prompts_cache[0])
    return {
        "cond_text_embeds_md5": _tensors_digest(cond.text_embeds),
        "uncond_text_embeds_md5": _tensors_digest(uncond.text_embeds),
        "sample_prompt_cache_file": Path(sd.sample_prompts_cache[0]).name,
    }


def _collect_inputs(process, dataset_dir: Path) -> Dict[str, Any]:
    sd = process.sd
    sampling_dit = getattr(sd, "_sampling_transformer", None)
    sampling_network = getattr(sd, "_sampling_network", None)
    record: Dict[str, Any] = {
        "dataset_files": sorted(p.name for p in dataset_dir.iterdir() if p.is_file()),
        "cache_files": {
            name: len(list((dataset_dir / name).glob("*.safetensors")))
            for name in CACHE_DIR_NAMES
        },
        "lora_up_max_abs_main": _lora_up_max_abs(process.network),
        "lora_up_max_abs_sampling": (
            None if sampling_network is None else _lora_up_max_abs(sampling_network)
        ),
        "vae_digest": _module_digest(sd.vae),
        "dit_digest": _module_digest(sampling_dit if sampling_dit is not None else sd.unet),
    }
    record.update(_prompt_embed_digests(sd))
    return record


def _install_baseline_probe(dataset_dir: Path, run_dir: Path) -> None:
    base_module = importlib.import_module("jobs.process.BaseSDTrainProcess")
    original_sample = base_module.BaseSDTrainProcess.sample

    def probed_sample(self, step=None, is_first=False):
        if step != 0:
            raise RuntimeError(f"expected the baseline sample at step 0, got step={step}")
        record = _collect_inputs(self, dataset_dir)
        original_sample(self, step, is_first)
        images = sorted((Path(self.save_root) / "samples").glob("*000000000*.png"))
        if len(images) != 1:
            raise RuntimeError(f"expected exactly one step-0 sample png, found {images}")
        shutil.copy2(images[0], run_dir / IMAGE_NAME)
        (run_dir / RESULT_NAME).write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise BaselineSampleDone

    base_module.BaseSDTrainProcess.sample = probed_sample


def _job_config(run_dir: Path, dataset_dir: Path) -> dict:
    model_path, sampling_path = _resolve_model_paths()
    config = _user_flowmatch_job_config(
        run_dir, dataset_dir, model_path, sampling_path, min_snr_gamma=0.0, steps=1
    )
    config["config"]["name"] = JOB_NAME
    process = config["config"]["process"][0]
    process["train"].update(
        skip_first_sample=False, disable_sampling=False, force_first_sample=False
    )
    process["datasets"][0].update(
        resolution=[512], caption_dropout_rate=0.1, caption_ext="txt"
    )
    process["sample"] = {
        "sampler": "flowmatch",
        "sample_every": 100,
        "width": 512,
        "height": 512,
        "samples": [{"prompt": SAMPLE_PROMPT}],
        "neg": "",
        "seed": SAMPLE_SEED,
        "walk_seed": False,
        "guidance_scale": 1,
        "sample_steps": 9,
        "format": "png",
    }
    return config


def _worker(dataset_dir: Path, run_dir: Path) -> None:
    from toolkit.job import run_job

    _install_baseline_probe(dataset_dir, run_dir)
    try:
        run_job(_job_config(run_dir, dataset_dir))
    except BaselineSampleDone:
        _log(f"[worker] baseline sample done -> {run_dir / IMAGE_NAME}")
        return
    raise RuntimeError("job finished without requesting the baseline sample at step 0")


# ---------------------------------------------------------------- orchestrator


def _dataset_stems() -> List[str]:
    stems = sorted(p.stem for p in SOURCE_DATASET_DIR.glob("*.png") if p.with_suffix(".txt").is_file())
    if len(stems) < 2:
        raise RuntimeError(f"need >= 2 png+txt pairs in {SOURCE_DATASET_DIR}, found {stems}")
    return stems


def _prepare_dataset(dataset_dir: Path, stems: List[str]) -> None:
    dataset_dir.mkdir(parents=True)
    for stem in stems:
        for suffix in (".png", ".txt"):
            shutil.copy2(SOURCE_DATASET_DIR / f"{stem}{suffix}", dataset_dir / f"{stem}{suffix}")
    leftovers = [name for name in CACHE_DIR_NAMES if (dataset_dir / name).exists()]
    if leftovers:
        raise RuntimeError(f"caches must be cleared before the run, found {leftovers}")


def _run_worker(label: str, stems: List[str], work_root: Path) -> Path:
    run_dir = work_root / label
    dataset_dir = run_dir / "dataset"
    _prepare_dataset(dataset_dir, stems)
    _log(f"\n=== run {label}: {len(stems)} pairs {stems[0]}..{stems[-1]} (caches cleared) ===")
    cmd = [
        sys.executable, "-m", "testing.zimage_step0_dataset_probe",
        "--worker", "--dataset-dir", str(dataset_dir), "--run-dir", str(run_dir),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return run_dir


def _image_diff(path_a: Path, path_b: Path) -> Dict[str, Any]:
    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return {"identical": False, "shape_mismatch": [a.shape, b.shape]}
    delta = np.abs(a - b)
    return {
        "identical": bool(not delta.any()),
        "max_abs_diff": int(delta.max()),
        "mean_abs_diff": float(delta.mean()),
    }


def _compare(name_a: str, dir_a: Path, name_b: str, dir_b: Path) -> Dict[str, Any]:
    rec_a = json.loads((dir_a / RESULT_NAME).read_text(encoding="utf-8"))
    rec_b = json.loads((dir_b / RESULT_NAME).read_text(encoding="utf-8"))
    compared = (
        "lora_up_max_abs_main", "lora_up_max_abs_sampling", "vae_digest", "dit_digest",
        "cond_text_embeds_md5", "uncond_text_embeds_md5",
    )
    inputs = {key: rec_a[key] == rec_b[key] for key in compared}
    image = _image_diff(dir_a / IMAGE_NAME, dir_b / IMAGE_NAME)
    _log(f"\n--- {name_a} vs {name_b} ---")
    for key, same in inputs.items():
        _log(f"  {key:28s} {'same' if same else 'DIFFERENT'}  ({rec_a[key]} | {rec_b[key]})")
    _log(f"  image: {image}")
    return {"inputs_equal": inputs, "image": image}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--no-control", action="store_true", help="skip the repeated run of dataset A")
    args = parser.parse_args()

    if args.worker:
        _worker(args.dataset_dir, args.run_dir)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    stems = _dataset_stems()
    base_stems, new_stems = stems[:-1], stems
    work_root = PROBE_ROOT / time.strftime("%Y%m%d_%H%M%S")
    _log(f"work root: {work_root}")

    dir_a = _run_worker("A", base_stems, work_root)
    dir_b = _run_worker("B_plus_one", new_stems, work_root)
    summary = {"A_vs_B": _compare("A", dir_a, "B_plus_one", dir_b)}
    if not args.no_control:
        dir_control = _run_worker("A_repeat", base_stems, work_root)
        summary["A_vs_A_repeat"] = _compare("A", dir_a, "A_repeat", dir_control)

    (work_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"\nsummary: {work_root / 'summary.json'}")


if __name__ == "__main__":
    main()
