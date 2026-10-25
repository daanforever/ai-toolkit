"""One imitation backward per microbatch for train.turbo_lora."""

import torch
import torch.nn.functional as F

from .turbo_schedule import get_turbo_sigmas_and_timesteps

_DROPOUT_ATTRS = ("rank_dropout", "module_dropout")


def format_turbo_lora_weight(weight: float) -> str:
    """Strip trailing zeros. 1.0 prints as 1."""
    return f"{float(weight):.12f}".rstrip("0").rstrip(".")


def run_imitation_pass(trainer, batch, microbatch_scale: float) -> torch.Tensor:
    """Residual MSE backward. Returns the detached unweighted MSE.

    target = (turbo_off + base_on - base_off).detach()
    Only turbo_on is in the autograd graph. Residency and dropout are restored
    on success and on raise.
    """
    sd = trainer.sd
    _require_imitation_inputs(sd, batch)
    saved_dropout = _save_dropout(sd.network, sd._sampling_network)
    try:
        _disable_dropout(saved_dropout)
        return _imitation_forwards(trainer, sd, batch, microbatch_scale)
    finally:
        _restore_dropout(saved_dropout)
        _restore_base_residency(sd)


def _require_imitation_inputs(sd, batch) -> None:
    if getattr(sd, "_sampling_transformer", None) is None:
        raise ValueError(
            "turbo_lora=true requires a loaded sampling transformer "
            "(set model.sampling_name_or_path to Z-Image-Turbo)."
        )
    if getattr(sd, "_sampling_network", None) is None:
        raise ValueError(
            "turbo_lora=true requires a loaded sampling network "
            "(shared LoRA on the sampling DiT)."
        )
    if getattr(batch, "prompt_embeds", None) is None:
        raise ValueError(
            "turbo_lora=true requires cached prompt embeds on the batch."
        )


def _imitation_forwards(trainer, sd, batch, microbatch_scale: float) -> torch.Tensor:
    latents = batch.latents
    embeds = batch.prompt_embeds
    x_t, timestep = _shared_noisy_latent(trainer, sd, latents)
    _set_inactive(sd.network)
    _set_inactive(sd._sampling_network)
    sd._train_on_turbo = False

    with torch.no_grad():
        base_off = _predict(sd, x_t, timestep, embeds)
    with torch.no_grad():
        with sd.network:
            base_on = _predict(sd, x_t, timestep, embeds)

    _set_inactive(sd.network)
    sd._train_on_turbo = True
    _set_inactive(sd._sampling_network)
    with torch.no_grad():
        turbo_off = _predict(sd, x_t, timestep, embeds)

    with sd._sampling_network:
        _set_inactive(sd.network)
        turbo_on = _predict(sd, x_t, timestep, embeds)
        target = (turbo_off + base_on - base_off).detach()
        residual_mse = F.mse_loss(turbo_on, target)
        weighted = float(trainer.train_config.turbo_lora_weight) * residual_mse
        trainer.accelerator.backward(weighted * microbatch_scale)
    return residual_mse.detach()


def _shared_noisy_latent(trainer, sd, latents):
    n_steps = int(trainer.train_config.turbo_prior_steps)
    _, centers = get_turbo_sigmas_and_timesteps(
        num_inference_steps=n_steps,
        use_dynamic_shifting=False,
    )
    slot = int(torch.randint(0, int(centers.shape[0]), (1,)).item())
    timestep = centers[slot].to(device=latents.device).expand(latents.shape[0]).contiguous()
    noise = torch.randn_like(latents)
    x_t = sd.add_noise(latents, noise, timestep)
    return x_t, timestep


def _predict(sd, x_t, timestep, embeds):
    return sd.get_noise_prediction(
        latent_model_input=x_t,
        timestep=timestep,
        text_embeddings=embeds,
    )


def _set_inactive(network) -> None:
    if network is not None and hasattr(network, "is_active"):
        network.is_active = False


def _dropout_carriers(*networks):
    carriers = []
    for network in networks:
        if network is None:
            continue
        if any(hasattr(network, name) for name in _DROPOUT_ATTRS):
            carriers.append(network)
        get_modules = getattr(network, "get_all_modules", None)
        if not callable(get_modules):
            continue
        for module in get_modules():
            if any(hasattr(module, name) for name in _DROPOUT_ATTRS):
                carriers.append(module)
    return carriers


def _save_dropout(*networks):
    saved = []
    for carrier in _dropout_carriers(*networks):
        values = {
            name: getattr(carrier, name)
            for name in _DROPOUT_ATTRS
            if hasattr(carrier, name)
        }
        saved.append((carrier, values))
    return saved


def _disable_dropout(saved) -> None:
    for carrier, values in saved:
        for name in values:
            setattr(carrier, name, None)


def _restore_dropout(saved) -> None:
    for carrier, values in saved:
        for name, value in values.items():
            setattr(carrier, name, value)


def _restore_base_residency(sd) -> None:
    sd._train_on_turbo = False
    parked = sd._park_sampling_dit_and_lora()
    if parked:
        sd._flush_cuda()
    sd._move_main_network(sd.device_torch)
