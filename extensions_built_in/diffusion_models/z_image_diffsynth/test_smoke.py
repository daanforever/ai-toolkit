"""
Smoke test for Z-Image DiffSynth: import, load model, prompt encoding, forward, generation pipeline.
Run from repo root. If venv exists, the script will use it automatically (no need to activate).

Paths: use env ZIMAGE_DIFFSYNTH_MODEL_PATH and optionally ZIMAGE_DIFFSYNTH_SAMPLING_PATH.
If unset, defaults from the plan are used (see DEFAULT_* below). Override via env if needed.

Regression: set ZIMAGE_DIFFSYNTH_TEST_QUANTIZE_TE=1 to run load with quantize_te=True
(checks that loader does not shadow quantize() with the bool param — "bool object is not callable").

Example (PowerShell, from repo root):
  python -m extensions_built_in.diffusion_models.z_image_diffsynth.test_smoke
"""

import io
import os
import sys
import tempfile

# Re-run with venv Python if venv exists and we're not already using it
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
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

# Default paths from the plan (HuggingFace snapshot layout)
DEFAULT_ZIMAGE_MODEL_PATH = (
    "e:/Backup/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/"
)
DEFAULT_ZIMAGE_SAMPLING_PATH = (
    "e:/Backup/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/0e36c2b379e66fa531d01cc531c44919e5f1c6fd/"
)

# repo root on path
TOOLKIT_ROOT = _REPO_ROOT
if TOOLKIT_ROOT not in sys.path:
    sys.path.insert(0, TOOLKIT_ROOT)


def _log(msg):
    """Print and flush so output appears immediately (helps when script fails mid-run)."""
    print(msg, flush=True)


class _TeeStderr:
    """Writes to both real stderr and a buffer. Use to capture loader/torch stderr during load_model()."""

    def __init__(self):
        self._buffer = io.StringIO()
        self._real = sys.stderr

    def write(self, s):
        self._real.write(s)
        self._real.flush()
        self._buffer.write(s)

    def flush(self):
        self._real.flush()
        self._buffer.flush()

    def getvalue(self):
        return self._buffer.getvalue()


def main():
    import traceback

    model_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_MODEL_PATH", "").strip() or DEFAULT_ZIMAGE_MODEL_PATH
    )
    _log(f"[paths] model_path={model_path!r}")
    if not model_path or not os.path.isdir(model_path):
        _log(
            "Z-Image model path is missing or not a directory. Set ZIMAGE_DIFFSYNTH_MODEL_PATH "
            "or use default (see DEFAULT_ZIMAGE_MODEL_PATH in this script)."
        )
        _log(f"  Current value: {model_path or '(empty)'}")
        sys.exit(1)
    _log("[paths] model_path OK")

    sampling_path = (
        os.environ.get("ZIMAGE_DIFFSYNTH_SAMPLING_PATH", "").strip()
        # or DEFAULT_ZIMAGE_SAMPLING_PATH
        or None
    )
    if sampling_path and not os.path.isdir(sampling_path):
        sampling_path = None
    _log(f"[paths] sampling_path={sampling_path or '(none)'}")

    _log("1. Import module and get model class ...")
    try:
        from toolkit.config_modules import ModelConfig
        from toolkit.util.get_model import get_model_class
        import torch
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    model_config_dict = {
        "name_or_path": model_path,
        "arch": "zimage_diffsynth",
        "quantize": False,
        "quantize_te": False,
    }
    if sampling_path:
        model_config_dict["sampling_name_or_path"] = sampling_path

    model_config = ModelConfig(**model_config_dict)
    ModelClass = get_model_class(model_config)
    if ModelClass.arch != "zimage_diffsynth":
        _log(f"Expected arch zimage_diffsynth, got {ModelClass.arch}")
        sys.exit(1)
    _log(f"   Model class: {ModelClass.__name__} (arch={ModelClass.arch})")
    _log("1. OK")

    _log("2. Create model and load_model() ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"   device={device}")
    tee = None
    old_stderr = None
    try:
        sd = ModelClass(device, model_config, dtype="bf16")
        _log("   ModelClass() OK, calling load_model() (stderr captured below) ...")
        tee = _TeeStderr()
        old_stderr = sys.stderr
        sys.stderr = tee
        sd.load_model()
    except Exception:
        if old_stderr is not None:
            sys.stderr = old_stderr
        if tee is not None:
            captured = tee.getvalue()
            if captured.strip():
                _log("[stderr during load_model]:")
                for line in captured.splitlines():
                    _log("  | " + line)
        _log("   FAILED in step 2 (create model / load_model):")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if old_stderr is not None:
            sys.stderr = old_stderr
    if tee is not None:
        captured = tee.getvalue()
        if captured.strip():
            _log("[stderr during load_model]:")
            for line in captured.splitlines():
                _log("  | " + line)
    _log("2. OK (Loaded.)")

    # Regression: load with quantize_te=True (ensures loader does not shadow quantize -> "bool not callable")
    if os.environ.get("ZIMAGE_DIFFSYNTH_TEST_QUANTIZE_TE", "").strip() == "1":
        _log("2b. Regression: load with quantize_te=True (no 'bool' callable error) ...")
        try:
            cfg_qt = {
                "name_or_path": model_path,
                "arch": "zimage_diffsynth",
                "quantize": False,
                "quantize_te": True,
            }
            if sampling_path:
                cfg_qt["sampling_name_or_path"] = sampling_path
            model_config_qt = ModelConfig(**cfg_qt)
            sd_qt = ModelClass(device, model_config_qt, dtype="bf16")
            sd_qt.load_model()
            _log("2b. OK")
        except Exception as e:
            _log(f"2b. FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        _log("2b. Skipped (set ZIMAGE_DIFFSYNTH_TEST_QUANTIZE_TE=1 to run)")

    _log("3. get_prompt_embeds ...")
    try:
        embeds = sd.get_prompt_embeds("a cat on a mat")
        assert embeds.text_embeds is not None
        te = embeds.text_embeds
        if isinstance(te, torch.Tensor):
            _log(f"   text_embeds shape: {te.shape}")
        else:
            _log(f"   text_embeds: {type(te)} (len={len(te) if hasattr(te, '__len__') else 'n/a'})")
    except Exception:
        _log("   FAILED in step 3 (get_prompt_embeds):")
        traceback.print_exc()
        sys.exit(1)
    _log("3. OK")

    _log("4. get_noise_prediction (dummy tensors) ...")
    try:
        B, C, H, W = 1, 16, 64, 64
        latent = torch.randn(B, C, H, W, device=device, dtype=sd.torch_dtype)
        timestep = torch.tensor([500], device=device, dtype=torch.float32)
        with torch.no_grad():
            pred = sd.get_noise_prediction(latent, timestep, embeds)
        _log(f"   noise_pred shape: {pred.shape}")
    except Exception:
        _log("   FAILED in step 4 (get_noise_prediction):")
        traceback.print_exc()
        sys.exit(1)
    _log("4. OK")

    _log("5. get_generation_pipeline ...")
    try:
        pipeline = sd.get_generation_pipeline()
        _log(f"   pipeline: {type(pipeline).__name__}")
    except Exception:
        _log("   FAILED in step 5 (get_generation_pipeline):")
        traceback.print_exc()
        sys.exit(1)
    _log("5. OK")

    _log("6. Optional: one step of sampling (single image) ...")
    try:
        from toolkit.config_modules import GenerateImageConfig
        gen_config = GenerateImageConfig(
            width=256, height=256, num_inference_steps=4,
            guidance_scale=1.0, prompt="a cat", negative_prompt="",
            output_folder=os.path.join(tempfile.gettempdir(), "z_image_diffsynth_smoke"),
            output_ext="png",
        )
        uncond = sd.get_prompt_embeds("")
        gen = torch.Generator(device=device).manual_seed(42)
        img = sd.generate_single_image(
            pipeline, gen_config, embeds, uncond, gen, {}
        )
        _log(f"   generated image: {type(img).__name__}")
    except Exception as e:
        _log(f"   sampling skipped or failed: {e}")
        traceback.print_exc()
    _log("6. OK (or skipped)")

    _log("Done.")


if __name__ == "__main__":
    main()
