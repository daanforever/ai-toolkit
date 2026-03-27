"""
Z-Image generation script based on the minimal ZImagePipeline example.
Config-driven: from_pretrained (or local components when transformer/ exists),
optional attention_backend, compile, CPU offload, quantization, LoRA.
Run from project root: python scripts/generate_samples.py [path/to/config.yaml]
Default config: temp/config.yaml (next to this script).
"""

import argparse
import gc
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import torch

try:
    import oyaml as yaml
except ImportError:
    import yaml

from diffusers import AutoencoderKL, ZImagePipeline
from diffusers.models.transformers import ZImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, Qwen3ForCausalLM


def _script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _repo_root():
    return os.path.dirname(_script_dir())


def _ensure_repo_in_path():
    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_in_path()
script_dir = _script_dir()
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import lora as lora_module
from toolkit.util.debug import is_debug_enabled, memory_debug, set_debug_config


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content = os.path.expandvars(content)
    return yaml.safe_load(content)


def _normalize_path(p: str) -> str:
    if not p:
        return p
    return os.path.abspath(os.path.expanduser(p))


def _get_torch_dtype(s: str):
    s = (s or "bfloat16").strip().lower()
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("float32", "fp32"):
        return torch.float32
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.bfloat16


def _flush_vram() -> None:
    """Release freed CUDA memory from PyTorch cache so process VRAM usage drops."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _quantize_module(model, model_cfg: dict, role: str) -> None:
    """Quantize and freeze a single module (transformer or text_encoder). Used right after load to free memory."""
    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from toolkit.util.quantize import get_qtype, quantize
    from optimum.quanto import freeze

    if role == "transformer" and model_cfg.get("quantize"):
        qtype_str = model_cfg.get("qtype", "qfloat8")
        quantize(model, weights=get_qtype(qtype_str))
        freeze(model)
    elif role == "text_encoder" and model_cfg.get("quantize_te"):
        qtype_te = model_cfg.get("qtype_te", model_cfg.get("qtype", "qfloat8"))
        quantize(model, weights=get_qtype(qtype_te))
        freeze(model)


def _load_pipeline_from_local(
    model_path: str,
    model_cfg: dict,
    torch_dtype,
    load_device: torch.device,
    loras: list,
    fuse_lora: bool,
):
    """
    Load Z-Image pipeline by components: transformer, tokenizer,
    text_encoder, vae, scheduler. For local snapshot with transformer/ subfolder.
    Transformer and text_encoder are placed on load_device. VAE and scheduler stay on default (moved with pipe later).
    When fuse_lora is False, LoRA is loaded into transformer here, then we quantize. When fuse_lora is True, we do not quantize here (done in main() after apply_loras_to_pipeline).
    """
    base_model_path = model_path
    te_folder = os.path.join(model_path, "text_encoder")
    if not os.path.isdir(te_folder):
        extras = model_cfg.get("extras_name_or_path")
        if extras:
            base_model_path = _normalize_path(os.path.expandvars(extras))
        else:
            base_model_path = model_path

    transformer_path = os.path.join(model_path, "transformer")
    if not os.path.isdir(transformer_path):
        raise FileNotFoundError(
            f"Transformer folder not found: {transformer_path}. "
            "Ensure model_path is a local directory with 'transformer' subfolder."
        )

    with memory_debug(print, "transformer load+quantize", kind="all"):
        if is_debug_enabled():
            print("  [debug] Before loading transformer...")
        print("  Loading transformer...")
        transformer = ZImageTransformer2DModel.from_pretrained(
            transformer_path,
            subfolder=None,
            torch_dtype=torch_dtype,
        )
        transformer = transformer.to(load_device)
        if is_debug_enabled():
            print("  [debug] After loading transformer.")

        if loras and not fuse_lora:
            if is_debug_enabled():
                print("  [debug] Before loading LoRA(s) into transformer...")
            print("  Loading LoRA(s) into transformer...")
            lora_module.load_loras_into_transformer(
                transformer,
                loras,
                normalize_path_func=_normalize_path,
                debug=is_debug_enabled(),
            )
            if is_debug_enabled():
                print("  [debug] After loading all LoRA(s).")

        if not fuse_lora:
            if is_debug_enabled():
                print("  [debug] Before quantizing transformer...")
            _quantize_module(transformer, model_cfg, "transformer")
            if is_debug_enabled():
                print("  [debug] After quantizing transformer.")
    if not fuse_lora and load_device.type == "cuda":
        _flush_vram()

    with memory_debug(print, "text_encoder load+quantize", kind="all"):
        if is_debug_enabled():
            print("  [debug] Before loading tokenizer and text encoder...")
        print("  Loading tokenizer and text encoder...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
        )
        text_encoder = text_encoder.to(load_device)
        if is_debug_enabled():
            print("  [debug] After loading text encoder.")
        if not fuse_lora:
            if is_debug_enabled():
                print("  [debug] Before quantizing text encoder...")
            _quantize_module(text_encoder, model_cfg, "text_encoder")
            if is_debug_enabled():
                print("  [debug] After quantizing text encoder.")
    if not fuse_lora and load_device.type == "cuda":
        _flush_vram()

    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        base_model_path,
        subfolder="vae",
        torch_dtype=torch_dtype,
    )

    print("  Loading scheduler...")
    scheduler_path = os.path.join(base_model_path, "scheduler")
    if os.path.isfile(os.path.join(scheduler_path, "scheduler_config.json")):
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_path,
            subfolder="scheduler",
        )
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
            use_dynamic_shifting=False,
        )

    pipe = ZImagePipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        transformer=transformer,
    )
    return pipe


def _apply_quantization(pipe, model_cfg: dict, device) -> None:
    """Apply quantization to transformer and optionally text_encoder using project toolkit."""
    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from toolkit.util.quantize import get_qtype, quantize
    from optimum.quanto import freeze

    qtype_str = model_cfg.get("qtype", "qfloat8")
    if model_cfg.get("quantize"):
        quantize(pipe.transformer, weights=get_qtype(qtype_str))
        freeze(pipe.transformer)
    if model_cfg.get("quantize_te") and getattr(pipe, "text_encoder", None) is not None:
        qtype_te = model_cfg.get("qtype_te", qtype_str)
        quantize(pipe.text_encoder, weights=get_qtype(qtype_te))
        freeze(pipe.text_encoder)


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image generation from YAML config (minimal ZImagePipeline example style)"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=os.path.join(_script_dir(), "config.yaml"),
        help="Path to YAML config (default: config.yaml next to script)",
    )
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)

    raw = _load_yaml(config_path)
    set_debug_config(
        SimpleNamespace(debug=raw.get("logging", {}).get("debug", False))
    )
    model_cfg = raw.get("model") or {}
    sample_cfg = raw.get("sample") or {}
    loras_adapters, fuse_lora = lora_module.parse_loras_config(raw.get("loras"))
    loras = lora_module.validate_loras(loras_adapters, normalize_path_func=_normalize_path)
    num_repeats = int(raw.get("num_repeats", 1))
    device_name = raw.get("device", "cuda")
    device = torch.device(device_name)
    load_mode = (model_cfg.get("load") or "vram").strip().lower()
    if load_mode not in ("vram", "ram"):
        load_mode = "vram"
    load_device = torch.device(device_name if load_mode == "vram" else "cpu")
    torch_dtype = _get_torch_dtype(raw.get("dtype", "bfloat16"))

    # Optional run options (from example comments)
    attention_backend = raw.get("attention_backend") or model_cfg.get("attention_backend")
    compile_transformer = raw.get("compile_transformer", False) or model_cfg.get("compile_transformer", False)
    cpu_offload = raw.get("cpu_offload", False) or model_cfg.get("cpu_offload", False)
    low_cpu_mem_usage = raw.get("low_cpu_mem_usage", False) if "low_cpu_mem_usage" in raw else model_cfg.get("low_cpu_mem_usage", False)

    model_path = model_cfg.get("name_or_path")
    if not model_path:
        print("model.name_or_path is required")
        sys.exit(1)
    model_path = os.path.expandvars(model_path)
    # Normalize only when it looks like a local path (absolute or existing dir)
    if os.path.isdir(model_path) or os.path.sep in model_path or (len(model_path) >= 2 and model_path[1:2] == ":"):
        model_path = _normalize_path(model_path)
    else:
        model_path = model_path.strip()

    output_dir = sample_cfg.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    prompts = sample_cfg.get("prompts") or []
    if not prompts:
        print("sample.prompts is empty")
        sys.exit(1)

    width = int(sample_cfg.get("width", 512))
    height = int(sample_cfg.get("height", 512))
    neg = sample_cfg.get("neg") or ""
    seed = int(sample_cfg.get("seed", -1))
    cfg = float(sample_cfg.get("cfg", sample_cfg.get("guidance_scale", 1.0)))
    steps = int(sample_cfg.get("steps", sample_cfg.get("sample_steps", 9)))
    ext = (sample_cfg.get("ext", "png") or "png").strip().lstrip(".")

    print("Loading model...")
    load_from_local = False
    transformer_path = os.path.join(model_path, "transformer") if os.path.isdir(model_path) else ""
    if transformer_path and os.path.isdir(transformer_path):
        load_from_local = True
        pipe = _load_pipeline_from_local(
            model_path, model_cfg, torch_dtype, load_device, loras=loras, fuse_lora=fuse_lora
        )
    else:
        pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    # Move to inference device only when we are not doing local+fuse_lora (lora and quantize must run on load_device first).
    if not (load_from_local and fuse_lora):
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

    if loras:
        lora_module.apply_loras_to_pipeline(
            pipe,
            loras,
            fuse_lora=fuse_lora,
            normalize_path_func=_normalize_path,
            debug=is_debug_enabled(),
        )

    # Quantization: for from_pretrained always here; for local+fuse_lora here (after fuse) on load_device; for local without fuse_lora already done in _load_pipeline_from_local.
    need_quantize = (load_from_local and fuse_lora) or (not load_from_local)
    if need_quantize and (model_cfg.get("quantize") or model_cfg.get("quantize_te")):
        print("Quantizing...")
        quantize_device = load_device if (load_from_local and fuse_lora) else device
        with memory_debug(print, "apply_quantization", kind="all"):
            _apply_quantization(pipe, model_cfg, quantize_device)
            if quantize_device.type == "cuda":
                _flush_vram()

    # For local+fuse_lora we deferred move: do it once after lora and quantize.
    if load_from_local and fuse_lora:
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

    if attention_backend:
        pipe.transformer.set_attention_backend(attention_backend)
    if compile_transformer:
        pipe.transformer.compile()

    image_index = 0
    total = num_repeats * len(prompts)
    print(f"Generating {total} image(s)...")
    for repeat in range(num_repeats):
        for i, prompt in enumerate(prompts):
            prompt = (prompt or "").strip()
            s = seed if seed >= 0 else None
            if seed >= 0 and (num_repeats > 1 or len(prompts) > 1):
                s = seed + repeat * len(prompts) + i
            generator = None
            if s is not None:
                generator = torch.Generator(device=device).manual_seed(s)
            out = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=cfg,
                negative_prompt=neg if neg else None,
                generator=generator,
            )
            image = out.images[0]
            now = datetime.now()
            base = now.strftime("%y%m%d%H%M%S")
            ms = now.microsecond // 1000
            filename = f"{base}{ms:03d}.{ext}"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"  Saved {filepath}")
            image_index += 1

    print("Done.")
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
