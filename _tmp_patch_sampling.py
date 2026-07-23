from pathlib import Path

path = Path("extensions_built_in/diffusion_models/z_image_diffsynth/sampling.py")
text = path.read_text(encoding="utf-8")
marker = "def get_generation_pipeline(sd_model):"
idx = text.index(marker)
new_fn = '''def get_generation_pipeline(sd_model):
    """Build pipeline for sd_model (ZImageDiffSynthModel). Uses sampling transformer if set.
    When loader=diffusers (or auto loaded as Diffusers), use ZImagePipeline with the
    Diffusers transformer. Otherwise use DiffSynth DiT + model_fn_z_image_turbo wrapper."""
    from toolkit.accelerator import unwrap_model
    from toolkit.paths import normalize_path

    sampling_is_diffusers = getattr(sd_model, "_sampling_is_diffusers", False)
    main_is_diffusers = getattr(sd_model, "_main_is_diffusers", False)
    sampling_transformer = getattr(sd_model, "_sampling_transformer", None)

    use_diffusers_pipeline = (
        (sampling_is_diffusers and sampling_transformer is not None)
        or (main_is_diffusers and sampling_transformer is None)
        or (main_is_diffusers and sampling_is_diffusers)
    )

    # Diffusers path: ZImagePipeline with Diffusers ZImageTransformer2DModel
    if use_diffusers_pipeline:
        from diffusers import ZImagePipeline

        if sampling_transformer is not None and (
            sampling_is_diffusers or main_is_diffusers
        ):
            tr_source = sampling_transformer
            pretrained_path = getattr(
                getattr(sd_model, "model_config", None), "sampling_name_or_path", None
            )
        else:
            tr_source = getattr(sd_model, "model", None)
            pretrained_path = getattr(
                getattr(sd_model, "model_config", None), "name_or_path", None
            )

        if pretrained_path:
            pretrained_path = normalize_path(pretrained_path)
            try:
                pipe = ZImagePipeline.from_pretrained(
                    pretrained_path,
                    torch_dtype=sd_model.torch_dtype,
                )
                tr = getattr(tr_source, "_inner_dit", getattr(tr_source, "dit", tr_source))
                pipe.transformer = unwrap_model(tr)
                use_dynamic_shifting = _resolve_use_dynamic_shifting_from_sd_model(
                    sd_model
                )
                if use_dynamic_shifting:
                    pipe.scheduler = CustomFlowMatchEulerDiscreteScheduler(
                        **build_scheduler_config(use_dynamic_shifting=True)
                    )
                return pipe
            except Exception:
                pass
        # Fallback: build from model components
        use_dynamic_shifting = _resolve_use_dynamic_shifting_from_sd_model(sd_model)
        scheduler = CustomFlowMatchEulerDiscreteScheduler(
            **build_scheduler_config(use_dynamic_shifting=use_dynamic_shifting)
        )
        vae = getattr(sd_model.vae, "vae_decoder", sd_model.vae)
        te = (
            sd_model.text_encoder[0]
            if isinstance(sd_model.text_encoder, list)
            else sd_model.text_encoder
        )
        tok = (
            sd_model.tokenizer[0]
            if isinstance(sd_model.tokenizer, list)
            else sd_model.tokenizer
        )
        if te is None:
            from toolkit.unloader import FakeTextEncoder

            te = FakeTextEncoder(
                device=sd_model.device_torch, dtype=sd_model.torch_dtype
            )
        tr = getattr(tr_source, "_inner_dit", getattr(tr_source, "dit", tr_source))
        return ZImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(te),
            tokenizer=tok,
            vae=unwrap_model(vae),
            transformer=unwrap_model(tr),
        )

    # DiffSynth path: ZImageDiT + model_fn_z_image_turbo
    sampling_dit = sampling_transformer
    raw_dit = getattr(sd_model, "_raw_dit", None)
    dit = sampling_dit if sampling_dit is not None else raw_dit
    if dit is None:
        dit = sd_model.model
    if isinstance(dit, torch.nn.Module) and "dit" in getattr(dit, "_modules", {}):
        dit = dit._modules["dit"]
    dit = getattr(dit, "_inner_dit", dit)
    vae = sd_model.vae
    vae_decoder = (
        vae
        if hasattr(vae, "decode")
        else (vae.vae_decoder if hasattr(vae, "vae_decoder") else vae)
    )
    tokenizer = (
        sd_model.tokenizer[0]
        if isinstance(sd_model.tokenizer, list)
        else sd_model.tokenizer
    )
    if isinstance(sd_model.text_encoder, list):
        text_encoder = (
            sd_model.text_encoder[0] if len(sd_model.text_encoder) > 0 else None
        )
    else:
        text_encoder = sd_model.text_encoder
    if text_encoder is None:
        from toolkit.unloader import FakeTextEncoder

        text_encoder = FakeTextEncoder(
            device=sd_model.device_torch, dtype=sd_model.torch_dtype
        )
    return ZImageDiffSynthPipelineWrapper(
        dit=unwrap_model(dit),
        vae=unwrap_model(vae_decoder),
        tokenizer=tokenizer,
        text_encoder=unwrap_model(text_encoder),
        device=sd_model.device_torch,
        dtype=sd_model.torch_dtype,
        use_dynamic_shifting=_resolve_use_dynamic_shifting_from_sd_model(sd_model),
    )
'''
path.write_text(text[:idx] + new_fn, encoding="utf-8")
print("updated", path)
