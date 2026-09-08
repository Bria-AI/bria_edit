"""Latent packing/unpacking helpers used by the training forward pass.

Extracted from `BriaPipeline` (`pipeline_bria_wan.py`) as plain functions.
They were the ONLY things training actually needed from that class: the
full inference pipeline (`Bria4EditPipelineWan` / `BriaPipeline`, subclassing
diffusers' `FluxPipeline`) is never instantiated by the training scripts --
`train_common.py` and `train_fibo_edit_standard.py` called these 5 methods
directly on the class object (`Bria4EditPipelineWan._pack_latents(...)`),
never on an instance. That meant the entire generation/sampling pipeline
(encode_prompt, the denoising loop, VAE decode, guidance) -- plus everything
it dragged in (`vae2_2.py`, `apg_utils.py`, `transformer_bria.py`) -- loaded
into the training process for nothing. See `README.md` in this directory.
"""

import torch


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents


def pack_latents_no_patch(latents, batch_size, num_channels_latents, height, width):
    latents = latents.permute(0, 2, 3, 1)
    latents = latents.reshape(batch_size, height * width, num_channels_latents)
    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


def unpack_latents_no_patch(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels)
    latents = latents.permute(0, 3, 1, 2)

    return latents


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)
