"""
Smoke test for Z-Image DiffSynth sampling config: verify SampleConfig supports
multiple prompts/samples (e.g. flowmatch sampler with sample_noised previews).

Run from repo root. If venv exists, the script will use it automatically
when executed as a module.
"""

import os
import sys

# Re-run with venv Python if venv exists and we're not already using it
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if sys.platform == "win32":
    _venv_python = os.path.join(_REPO_ROOT, "venv", "Scripts", "python.exe")
else:
    _venv_python = os.path.join(_REPO_ROOT, "venv", "bin", "python")
if os.path.isfile(_venv_python):
    _current = os.path.realpath(sys.executable)
    _venv_real = os.path.realpath(_venv_python)
    if _current != _venv_real:
        os.execv(_venv_python, [_venv_python] + sys.argv[1:])

# repo root on path
TOOLKIT_ROOT = _REPO_ROOT
if TOOLKIT_ROOT not in sys.path:
    sys.path.insert(0, TOOLKIT_ROOT)


def _log(msg: str) -> None:
    """Print and flush so output appears immediately."""
    print(msg, flush=True)


def _test_sample_config_multiple_prompts() -> None:
    """
    Construct a SampleConfig equivalent to the provided YAML snippet:

      sample:
        sample_noised: true
        sampler: "flowmatch"
        sample_every: 50
        width: 1024
        height: 768
        samples:
          - prompt: "cat"
          - prompt: "dog"
          - prompt: "fish"

    and verify that prompts and core fields are wired correctly.
    """
    from toolkit.config_modules import SampleConfig

    _log("Sampling smoke: constructing SampleConfig with multiple prompts ...")
    sample_cfg = SampleConfig(
        sampler="flowmatch",
        sample_noised=True,
        sample_every=50,
        width=1024,
        height=768,
        samples=[
            {"prompt": "cat"},
            {"prompt": "dog"},
            {"prompt": "fish"},
        ],
    )

    # Core fields
    assert sample_cfg.sampler == "flowmatch", f"unexpected sampler: {sample_cfg.sampler!r}"
    assert sample_cfg.sample_noised is True, f"sample_noised should be True, got {sample_cfg.sample_noised!r}"
    assert sample_cfg.sample_every == 50, f"unexpected sample_every: {sample_cfg.sample_every!r}"
    assert sample_cfg.width == 1024, f"unexpected width: {sample_cfg.width!r}"
    assert sample_cfg.height == 768, f"unexpected height: {sample_cfg.height!r}"

    # Samples / prompts
    assert sample_cfg.samples is not None, "sample_cfg.samples must not be None"
    assert len(sample_cfg.samples) == 3, f"expected 3 samples, got {len(sample_cfg.samples)}"
    prompts = sample_cfg.prompts
    assert prompts == ["cat", "dog", "fish"], f"unexpected prompts: {prompts!r}"

    _log("Sampling smoke: SampleConfig multiple-prompts check OK.")


def _test_batch_sampling_device_moves() -> None:
    """
    Load Z-Image DiffSynth with a sampling transformer and run batch sampling
    for multiple prompts, asserting that we do not trigger per-prompt CPU↔GPU
    moves. There must be exactly one batch enter/exit for sampling:
      - one \"enable\" log before generation
      - one \"restore\" log after generation
    and no standalone-per-prompt sampling logs.
    """
    import tempfile
    from types import SimpleNamespace

    import torch
    from toolkit.util.debug import set_debug_config
    from toolkit.config_modules import ModelConfig, GenerateImageConfig
    from toolkit.util.get_model import get_model_class
    from extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke import (
        DEFAULT_ZIMAGE_MODEL_PATH,
        DEFAULT_ZIMAGE_SAMPLING_PATH,
    )

    # Enable debug so our is_debug_enabled()-guarded logs fire.
    set_debug_config(SimpleNamespace(debug=True))

    model_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip()
        or DEFAULT_ZIMAGE_MODEL_PATH
    )
    sampling_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_SAMPLING_PATH", "").strip()
        or DEFAULT_ZIMAGE_SAMPLING_PATH
        or None
    )

    _log(f"[batch] model_path={model_path!r}")
    if not model_path or not os.path.isdir(model_path):
        _log("[batch] model path missing or invalid; skipping device-move test.")
        return
    if sampling_path and not os.path.isdir(sampling_path):
        sampling_path = None
    _log(f"[batch] sampling_path={sampling_path or '(none)'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        _log("[batch] CUDA not available; skipping device-move test.")
        return

    model_config_dict = {
        "name_or_path": model_path,
        "arch": "zimage_diffsynth",
        "quantize": True,
        "quantize_te": True,
    }
    if sampling_path:
        model_config_dict["sampling_name_or_path"] = sampling_path

    model_config = ModelConfig(**model_config_dict)
    ModelClass = get_model_class(model_config)

    _log("[batch] creating model and calling load_model() ...")
    sd = ModelClass(device, model_config, dtype="bf16")
    sd.load_model()

    if getattr(sd, "_sampling_transformer", None) is None:
        _log("[batch] no sampling transformer present; skipping device-move test.")
        return

    # Capture debug/status logs emitted via print_and_status_update.
    captured_logs: list[str] = []
    orig_print = sd.print_and_status_update

    def _wrapped_print(msg):
        text = str(msg)
        captured_logs.append(text)
        return orig_print(text)

    sd.print_and_status_update = _wrapped_print  # type: ignore[assignment]

    try:
        _log("[batch] building generation pipeline ...")
        pipeline = sd.get_generation_pipeline()
        _log(f"[batch] pipeline: {type(pipeline).__name__}")

        tmp_out = os.path.join(
            tempfile.gettempdir(), "z_image_diffsynth_smoke_batch"
        )

        _log("[batch] running generate_images for 3 prompts ...")
        gen_configs = [
            GenerateImageConfig(
                width=1024,
                height=768,
                num_inference_steps=2,
                guidance_scale=1.0,
                prompt=p,
                negative_prompt="",
                output_folder=tmp_out,
                output_ext="png",
            )
            for p in ("cat", "dog", "fish")
        ]
        # sampler name here is mostly informational; the model always uses flow-match
        # style sampling internally.
        sd.generate_images(gen_configs, sampler="flowmatch")
    finally:
        # Restore original printer so we don't affect other callers.
        sd.print_and_status_update = orig_print  # type: ignore[assignment]

    # Analyse captured logs for device-move patterns.
    standalone_begin = (
        "[zimage_diffsynth] standalone sampling: moving main transformer to "
        "CPU and sampling transformer to GPU"
    )
    standalone_end = (
        "[zimage_diffsynth] standalone sampling: restoring main "
        "transformer to GPU and sampling transformer to CPU"
    )
    batch_begin = (
        "[zimage_diffsynth] batch generate: enabling sampling transformer "
        "on GPU and using sampling network"
    )
    batch_end = (
        "[zimage_diffsynth] batch generate: restoring main "
        "transformer to GPU and moving sampling transformer to CPU"
    )

    standalone_begin_count = sum(1 for l in captured_logs if standalone_begin in l)
    standalone_end_count = sum(1 for l in captured_logs if standalone_end in l)
    batch_begin_count = sum(1 for l in captured_logs if batch_begin in l)
    batch_end_count = sum(1 for l in captured_logs if batch_end in l)

    _log(
        f"[batch] standalone_begin={standalone_begin_count}, "
        f"standalone_end={standalone_end_count}, "
        f"batch_begin={batch_begin_count}, batch_end={batch_end_count}"
    )

    # There must be no per-prompt standalone sampling moves.
    assert (
        standalone_begin_count == 0 and standalone_end_count == 0
    ), "Expected no standalone sampling device moves during batch generate_images"

    # And exactly one pair of batch enter/exit logs for the whole multi-prompt run.
    assert (
        batch_begin_count == 1 and batch_end_count == 1
    ), f"Expected exactly one batch enter/exit, got begin={batch_begin_count}, end={batch_end_count}"

    _log("[batch] Device-move pattern for multi-prompt batch sampling OK.")


def main() -> None:
    _log("Z-Image DiffSynth sampling smoke test (SampleConfig multi-prompt + batch device moves) ...")
    try:
        _test_sample_config_multiple_prompts()
        _test_batch_sampling_device_moves()
    except AssertionError as e:
        _log(f"FAILED: assertion error in sampling smoke test: {e}")
        raise SystemExit(1)
    except Exception as e:  # pragma: no cover - defensive
        _log(f"FAILED: unexpected error in sampling smoke test: {e}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
    _log("Done.")


if __name__ == "__main__":
    main()

