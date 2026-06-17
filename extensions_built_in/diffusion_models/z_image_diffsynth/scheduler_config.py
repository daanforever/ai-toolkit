"""Shared FlowMatch scheduler config for z_image_diffsynth training and sampling."""

STATIC_SHIFT = 3.0

DYNAMIC_SHIFT_DEFAULTS = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
}


def build_scheduler_config(use_dynamic_shifting: bool = False) -> dict:
    cfg = {
        "num_train_timesteps": 1000,
        "use_dynamic_shifting": use_dynamic_shifting,
        "shift": STATIC_SHIFT,
    }
    if use_dynamic_shifting:
        cfg.update(DYNAMIC_SHIFT_DEFAULTS)
    return cfg
