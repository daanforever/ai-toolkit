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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae_encoder, "encode"):
            # diffusers AutoencoderKL
            return self.vae_encoder.encode(x).latent_dist.sample()
        return self.vae_encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae_decoder, "decode"):
            return self.vae_decoder.decode(z).sample
        return self.vae_decoder(z)
