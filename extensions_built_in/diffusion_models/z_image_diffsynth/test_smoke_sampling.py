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


def _test_move_sampling_transformer_need_move() -> None:
    """
    When shared LoRA is already on GPU but base is on CPU, _move_sampling_transformer
    must still move base to the target device (not skip via next(parameters())).
    """
    import torch
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
        _DiTUnetWrapper,
    )

    if not torch.cuda.is_available():
        _log("[need_move] CUDA not available; skipping.")
        return

    class _InnerDit(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frozen_linear = torch.nn.Linear(4, 4, bias=False)
            self.frozen_linear.weight.requires_grad_(False)
            self.lora_linear = torch.nn.Linear(4, 4, bias=False)
            self.lora_linear.weight.requires_grad_(True)

    inner = _InnerDit()
    inner.frozen_linear.to("cpu")
    inner.lora_linear.to("cuda")
    wrapper = _DiTUnetWrapper(inner)

    sd = ZImageDiffSynthModel.__new__(ZImageDiffSynthModel)
    sd.print_and_status_update = lambda *_args, **_kwargs: None
    sd._sampling_transformer = wrapper
    sd.device_torch = torch.device("cuda")

    base = ZImageDiffSynthModel._first_frozen_base_param(wrapper)
    assert base is not None and base.device.type == "cpu"

    sd._move_sampling_transformer("cuda")
    assert base.device.type == "cuda", "base should move to GPU when LoRA already on GPU"

    sd._move_sampling_transformer("cpu")
    assert base.device.type == "cpu", "base should move back to CPU"

    _log("[need_move] _move_sampling_transformer base-param device check OK.")


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
    from extensions_built_in.diffusion_models.z_image_diffsynth.model import (
        ZImageDiffSynthModel,
    )
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

    gpu_base_devices: list[str] = []
    st = sd._sampling_transformer
    _orig_st_to = st.to

    def _spy_st_to(*args, **kwargs):
        result = _orig_st_to(*args, **kwargs)
        dev = args[0] if args else kwargs.get("device")
        if dev is not None:
            target = dev if isinstance(dev, torch.device) else torch.device(dev)
            if target.type == "cuda":
                base_p = ZImageDiffSynthModel._first_frozen_base_param(st)
                if base_p is not None:
                    gpu_base_devices.append(base_p.device.type)
        return result

    st.to = _spy_st_to  # type: ignore[method-assign]

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

        base_after = ZImageDiffSynthModel._first_frozen_base_param(st)
        assert base_after is not None, "expected frozen base param on sampling transformer"
        assert (
            base_after.device.type == "cpu"
        ), f"after sampling, base should be on CPU, got {base_after.device}"
        assert gpu_base_devices, "expected at least one GPU move during sampling"
        assert (
            gpu_base_devices[-1] == "cuda"
        ), f"during sampling, base should be on GPU, saw {gpu_base_devices!r}"
    finally:
        st.to = _orig_st_to  # type: ignore[method-assign]
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


def _test_share_parameters_edge_cases() -> None:
    """
    Exercise LoRASpecialNetwork.share_parameters_with for edge cases:
    - LoRA parameter and buffer sharing between two networks
    - full_train_in_out parameter sharing (unet_conv_in/out)
    - mismatch detection on lora_name.
    """
    import torch
    from toolkit.lora_special import LoRASpecialNetwork

    class DummyTextEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

    # Create a dummy UNet-like module whose class name matches what
    # LoRASpecialNetwork expects ("UNet2DConditionModel").
    class UNet2DConditionModel(torch.nn.Module):  # type: ignore[override]
        def __init__(self) -> None:
            super().__init__()
            self.conv_in = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)
            self.conv_out = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)
            # Put LoRA-compatible modules inside a submodule whose class name
            # matches LINEAR_MODULES/CONV_MODULES expectations so that
            # create_modules() in LoRASpecialNetwork will pick them up.
            class LoRACompatibleLinear(torch.nn.Linear):
                pass

            class LoRACompatibleConv(torch.nn.Conv2d):
                pass

            self.block = torch.nn.Module()
            self.block.linear = LoRACompatibleLinear(4, 4)
            self.block.conv = LoRACompatibleConv(4, 4, kernel_size=1)

    _log("[share] building two LoRASpecialNetwork instances for edge-case tests ...")
    text_enc = DummyTextEncoder()
    unet_a = UNet2DConditionModel()
    unet_b = UNet2DConditionModel()

    net_a = LoRASpecialNetwork(
        text_encoder=text_enc,
        unet=unet_a,
        train_text_encoder=False,
        train_unet=True,
        lora_dim=2,
        alpha=1.0,
        full_train_in_out=True,
        target_lin_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE,
        target_conv_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3,
    )
    net_b = LoRASpecialNetwork(
        text_encoder=text_enc,
        unet=unet_b,
        train_text_encoder=False,
        train_unet=True,
        lora_dim=2,
        alpha=1.0,
        full_train_in_out=True,
        target_lin_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE,
        target_conv_modules=LoRASpecialNetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3,
    )

    assert net_a.unet_loras, "[share] expected at least one unet LoRA module"
    assert len(net_a.unet_loras) == len(
        net_b.unet_loras
    ), "[share] unet_loras length mismatch between test networks"

    # 1) Base case: after sharing, LoRA parameters and buffers are the same objects.
    lora_a = net_a.unet_loras[0]
    lora_b = net_b.unet_loras[0]
    # Sanity: parameters and buffers differ before sharing.
    assert lora_a.lora_down.weight is not lora_b.lora_down.weight
    assert getattr(lora_a, "alpha") is not getattr(
        lora_b, "alpha"
    ), "[share] alpha buffers unexpectedly shared before share_parameters_with"

    net_b.share_parameters_with(net_a)

    assert (
        lora_a.lora_down.weight is lora_b.lora_down.weight
    ), "[share] lora_down.weight not shared after share_parameters_with"
    assert (
        getattr(lora_a, "alpha") is getattr(lora_b, "alpha")
    ), "[share] alpha buffer not shared after share_parameters_with"

    # 2) full_train_in_out: conv_in/conv_out should also share parameters.
    assert hasattr(net_a, "unet_conv_in") and hasattr(
        net_a, "unet_conv_out"
    ), "[share] net_a missing full_train_in_out conv modules"
    assert hasattr(net_b, "unet_conv_in") and hasattr(
        net_b, "unet_conv_out"
    ), "[share] net_b missing full_train_in_out conv modules"

    assert (
        net_a.unet_conv_in.weight is net_b.unet_conv_in.weight
    ), "[share] unet_conv_in weights not shared after share_parameters_with"
    assert (
        net_a.unet_conv_out.weight is net_b.unet_conv_out.weight
    ), "[share] unet_conv_out weights not shared after share_parameters_with"

    # 3) Mismatch detection: differing lora_name should trigger an assertion.
    orig_name = net_b.unet_loras[0].lora_name
    net_b.unet_loras[0].lora_name = orig_name + "_mismatch"
    try:
        raised = False
        net_b.share_parameters_with(net_a)
    except AssertionError as e:
        raised = True
        msg = str(e)
        assert "lora name mismatch" in msg, f"[share] unexpected assertion message: {msg!r}"
    finally:
        net_b.unet_loras[0].lora_name = orig_name

    assert raised, "[share] expected AssertionError on lora name mismatch"

    _log("[share] share_parameters_with edge-case tests OK.")


def main() -> None:
    _log("Z-Image DiffSynth sampling smoke test (SampleConfig multi-prompt + batch device moves) ...")
    try:
        _test_sample_config_multiple_prompts()
        _test_move_sampling_transformer_need_move()
        _test_batch_sampling_device_moves()
        _test_share_parameters_edge_cases()
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

