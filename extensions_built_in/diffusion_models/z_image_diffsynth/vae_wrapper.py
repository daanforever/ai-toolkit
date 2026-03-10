# VAE wrapper: single object with .encode() and .decode().

from typing import Optional
import torch


class DiffSynthVAEWrapper(torch.nn.Module):
    """Wraps vae_encoder and vae_decoder (or a single AutoencoderKL) as one object with .encode()/.decode().
    Subclasses nn.Module so .to(), .parameters(), .train(), .eval(), etc. work for the trainer and pipeline.
    """

    def __init__(self, vae_encoder, vae_decoder=None):
        super().__init__()
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder if vae_decoder is not None else vae_encoder

        # Expose .config like a regular VAE so BaseModel / SDTrainer code that
        # expects self.vae.config[...] continues to work.
        inner_config = getattr(self.vae_encoder, "config", None)
        if inner_config is not None:
            # Most diffusers VAEs already have a config object that supports
            # attribute and/or item access; just reuse it.
            self.config = inner_config
        else:
            # Fallback minimal config with sane defaults; mainly used to keep
            # encode/decode_latents and bucket divisibility from crashing.
            class _Config(dict):
                def __getattr__(self, key):
                    try:
                        return self[key]
                    except KeyError as e:
                        raise AttributeError(key) from e

            self.config = _Config(
                block_out_channels=(4,),
                scaling_factor=1.0,
                shift_factor=0.0,
            )

    @property
    def device(self) -> torch.device:
        """Device of the first parameter (expected by base_model save_device_state / set_device_state)."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the first parameter (expected by SDTrainer train_single_accumulation: vae.dtype vs vae_torch_dtype)."""
        return next(self.parameters()).dtype

    def encode(self, x: torch.Tensor):
        """Mirror diffusers AutoencoderKL.encode API where possible."""
        if hasattr(self.vae_encoder, "encode"):
            # Return the same object (e.g. AutoencoderKLOutput) so callers that
            # expect .latent_dist.sample() keep working.
            return self.vae_encoder.encode(x)
        return self.vae_encoder(x)

    def decode(self, z: torch.Tensor):
        """Mirror diffusers AutoencoderKL.decode API where possible."""
        if hasattr(self.vae_decoder, "decode"):
            # Return the same object (e.g. DecoderOutput) so callers that
            # expect .sample keep working.
            return self.vae_decoder.decode(z)
        return self.vae_decoder(z)
