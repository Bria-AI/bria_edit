"""
Common utilities for FIBO Edit training scripts.

Shared infrastructure extracted from train_fibo_edit_unified.py.
Imported by mode-specific training scripts (train_fibo_edit_standard.py, etc.)
"""

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import diffusers
import torch
from accelerate.logging import get_logger
from model.latent_packing import pack_latents, pack_latents_no_patch, prepare_latent_image_ids, unpack_latents, unpack_latents_no_patch

logger = get_logger(__name__, log_level="INFO")


# =============================================================================
# Mode 3: Auto-Discovery Helpers
# =============================================================================


def parse_data_config(data_config: str) -> List[Dict]:
    """Parse data_config string into list of dataset dicts.

    Format: "name:gpus:batch,name:gpus:batch,..."
    Example: "A:32:1" or "TEXT:16:1,IMG:16:2"
    """
    if not data_config:
        return []

    datasets = []
    for part in data_config.split(","):
        parts = part.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid data_config format: '{part}'. Expected 'name:gpus:batch'")
        datasets.append({"name": parts[0], "n_gpus": int(parts[1]), "batch_size": int(parts[2])})
    return datasets


def discover_resolutions(mount_path: str, min_res: int = 0, max_res: int = 0) -> List[Tuple[int, int]]:
    """Discover available resolutions by scanning mounted directory.

    Looks for directories matching pattern {width}x{height}.
    Returns list of (width, height) tuples, sorted.

    Optionally filters by nominal resolution = round(sqrt(W*H)) — the area-equivalent
    square side, which cleanly labels the area-preserving AR families (~256/512/1024)
    even though individual sides overlap across families. 0 = unbounded.
    A bucket is kept iff (min_res==0 or res>=min_res) and (max_res==0 or res<=max_res).
    """
    resolutions = []
    if not os.path.exists(mount_path):
        raise FileNotFoundError(f"Mount path not found: {mount_path}")

    for entry in os.listdir(mount_path):
        entry_path = os.path.join(mount_path, entry)
        if os.path.isdir(entry_path) and "x" in entry:
            try:
                width_str, height_str = entry.split("x")
                width, height = int(width_str), int(height_str)
            except ValueError:
                continue  # Skip entries that don't match WxH pattern
            res = round((width * height) ** 0.5)
            if min_res and res < min_res:
                continue
            if max_res and res > max_res:
                continue
            resolutions.append((width, height))

    return sorted(resolutions)


def allocate_gpus_to_resolutions(resolutions: List[Tuple[int, int]], total_gpus: int) -> List[int]:
    """Allocate GPUs fairly across resolutions.

    Returns list of GPU counts per resolution.
    """
    n = len(resolutions)
    if n == 0:
        raise ValueError("No resolutions found to allocate GPUs")

    base = total_gpus // n
    remainder = total_gpus % n
    return [base + (1 if i < remainder else 0) for i in range(n)]


def auto_discover_data(
    rank: int,
    data_config: str,
    attach_structured_captions: bool = True,
) -> Tuple[str, int, int, int, bool]:
    """Auto-discover data configuration for mode 3.

    Similar to json_to_data but discovers resolutions from mounted directories.

    Returns:
        Tuple of (training_dir, width, height, batch_size, is_multi_res)
        Note: is_multi_res is always False for auto-discovery mode (fixed resolutions).
    """
    datasets = parse_data_config(data_config)
    if not datasets:
        raise ValueError("data_config is empty")

    # Build allocation map similar to json_to_data
    gpu_cumsum = 0
    curr_dataset = None

    for ds in datasets:
        if rank < gpu_cumsum + ds["n_gpus"]:
            curr_dataset = ds
            break
        gpu_cumsum += ds["n_gpus"]

    if curr_dataset is None:
        raise ValueError(f"Rank {rank} not found in data_config: {data_config}")

    ds_name = curr_dataset["name"]
    batch_size = curr_dataset["batch_size"]

    # Get mount path from environment variable (SageMaker sets SM_CHANNEL_DATASET_X)
    env_key = f"SM_CHANNEL_DATASET_{ds_name}"
    mount_path = os.environ.get(env_key)
    if not mount_path:
        raise ValueError(f"Environment variable {env_key} not set")

    # Discover resolutions
    resolutions = discover_resolutions(mount_path)
    if not resolutions:
        raise ValueError(f"No resolution directories found in {mount_path}")

    # Allocate GPUs to resolutions
    gpu_allocation = allocate_gpus_to_resolutions(resolutions, curr_dataset["n_gpus"])

    # Find which resolution this rank should use
    rank_in_dataset = rank - gpu_cumsum
    current_sum = 0
    for i, (w, h) in enumerate(resolutions):
        current_sum += gpu_allocation[i]
        if rank_in_dataset < current_sum:
            width, height = w, h
            training_dir = f"{mount_path.rstrip('/')}/{w}x{h}"
            if attach_structured_captions:
                training_dir += "/structured-captions"
            return training_dir, width, height, batch_size, False

    # Should never reach here
    raise ValueError(f"Failed to allocate rank {rank} to a resolution")


def auto_discover_data_interleaved(
    rank: int,
    data_config: str,
    attach_structured_captions: bool = True,
    min_res: int = 0,
    max_res: int = 0,
    data_paths: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], int, str]:
    """Return the training dirs for the dataset (channel) this rank belongs to.

    Resolution-sharded channels return ALL <W>x<H> dirs (RandomMix'd by dataset_factory).
    Flat channels (no <W>x<H> subdirs, e.g. multi_ref) return the single structured-captions dir.
    Unlike auto_discover_data() which assigns each rank to ONE resolution.

    Returns:
        Tuple of (training_dirs_list, batch_size, ds_name)
    """
    datasets = parse_data_config(data_config)
    if not datasets:
        raise ValueError("data_config is empty")

    # Find which dataset this rank belongs to
    gpu_cumsum = 0
    curr_dataset = None

    for ds in datasets:
        if rank < gpu_cumsum + ds["n_gpus"]:
            curr_dataset = ds
            break
        gpu_cumsum += ds["n_gpus"]

    if curr_dataset is None:
        raise ValueError(f"Rank {rank} not found in data_config: {data_config}")

    ds_name = curr_dataset["name"]
    batch_size = curr_dataset["batch_size"]

    # Resolve mount path: config.data_paths first (committed + persisted in train_config.yaml),
    # then the legacy SM_CHANNEL_DATASET_<NAME> env var as fallback.
    mount_path = (data_paths or {}).get(ds_name) or os.environ.get(f"SM_CHANNEL_DATASET_{ds_name}")
    if not mount_path:
        raise ValueError(
            f"No data path for '{ds_name}': set data_paths['{ds_name}'] in the config "
            f"or the SM_CHANNEL_DATASET_{ds_name} env var"
        )

    # Discover all resolutions (optionally filtered by nominal resolution to one AR family)
    resolutions = discover_resolutions(mount_path, min_res=min_res, max_res=max_res)

    if resolutions:
        # Resolution-sharded channel: one training dir per <W>x<H> bucket (RandomMix across them).
        training_dirs = []
        for w, h in resolutions:
            d = f"{mount_path.rstrip('/')}/{w}x{h}"
            if attach_structured_captions:
                d += "/structured-captions"
            training_dirs.append(d)
    else:
        # Flat channel (e.g. multi_ref): tars live directly under structured-captions/ with mixed
        # per-sample resolution (no <W>x<H> sharding). Load that one dir as a single stream — the
        # min_res/max_res family filter does not apply. (One training_dir -> dataset_factory.build
        # uses the direct load_dataset_from_tars branch, not RandomMix.)
        flat_dir = f"{mount_path.rstrip('/')}/structured-captions" if attach_structured_captions else mount_path.rstrip("/")
        if not os.path.isdir(flat_dir):
            raise ValueError(
                f"No <W>x<H> resolution dirs found in {mount_path} and no flat '{flat_dir}' dir "
                f"either (min_res={min_res}, max_res={max_res})"
            )
        training_dirs = [flat_dir]

    return training_dirs, batch_size, ds_name


# =============================================================================
# Reproducibility Helpers
# =============================================================================


def get_git_info() -> Dict[str, str]:
    """Get git commit hash, branch, and diff summary for reproducibility."""
    info = {"git_commit": "unknown", "git_branch": "unknown", "git_diff_summary": ""}
    try:
        info["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
        info["git_branch"] = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        diff_output = subprocess.check_output(["git", "diff", "--stat"], stderr=subprocess.DEVNULL).decode().strip()
        info["git_diff_summary"] = diff_output[:1000] if diff_output else "clean"
    except Exception:
        pass
    return info


def get_system_info() -> Dict[str, Any]:
    """Get system info (CUDA, GPU, versions) for reproducibility."""
    info = {
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "gpu_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    return info


# =============================================================================
# Text Encoder Layer Preparation
# =============================================================================


def prepare_text_encoder_layers(
    text_encoder_layers: List[torch.Tensor],
    null_conditioning_layers: Optional[List[torch.Tensor]],
    total_num_layers: int,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """Trim or extend text encoder layers to match transformer layers."""
    if len(text_encoder_layers) >= total_num_layers:
        text_encoder_layers = text_encoder_layers[len(text_encoder_layers) - total_num_layers :]
        if null_conditioning_layers is not None:
            null_conditioning_layers = null_conditioning_layers[len(null_conditioning_layers) - total_num_layers :]
    else:
        text_encoder_layers = text_encoder_layers + [text_encoder_layers[-1]] * (
            total_num_layers - len(text_encoder_layers)
        )
        if null_conditioning_layers is not None:
            null_conditioning_layers = null_conditioning_layers + [null_conditioning_layers[-1]] * (
                total_num_layers - len(null_conditioning_layers)
            )

    return text_encoder_layers, null_conditioning_layers


# =============================================================================
# Latent Preparation
# =============================================================================


def prepare_latents(
    noisy_latents: torch.Tensor,
    context_latents: Optional[torch.Tensor],
    vae_scale_factor: int,
    height: int,
    width: int,
    device: torch.device,
    do_patching: int = 0,
) -> Dict[str, Any]:
    """Pack latents and prepare image IDs for flash attention path.

    Note: This is only used for flash attention (bsz=1). Varlen mode handles
    latent packing differently in prepare_forward_inputs_varlen.
    """
    num_channels_latents = noisy_latents.shape[1]
    latent_height = int(height) // vae_scale_factor
    latent_width = int(width) // vae_scale_factor

    # Target latent packing (differs by do_patching).
    if do_patching == 0:
        patched_noisy_latents = pack_latents_no_patch(
            latents=noisy_latents,
            batch_size=noisy_latents.shape[0],
            num_channels_latents=num_channels_latents,
            height=latent_height,
            width=latent_width,
        )
        patched_latent_image_ids = prepare_latent_image_ids(
            batch_size=noisy_latents.shape[0],
            height=latent_height,
            width=latent_width,
            device=device,
            dtype=noisy_latents.dtype,
        )
    else:
        patched_noisy_latents = pack_latents(
            latents=noisy_latents,
            batch_size=noisy_latents.shape[0],
            num_channels_latents=num_channels_latents,
            height=latent_height,
            width=latent_width,
        )
        patched_latent_image_ids = prepare_latent_image_ids(
            batch_size=noisy_latents.shape[0],
            height=latent_height // 2,
            width=latent_width // 2,
            device=device,
            dtype=noisy_latents.dtype,
        )

    # Context packing: N ordered contexts (pos-ids 1..N), each at its OWN latent dims so
    # mixed-size refs work. context_latents may be a single tensor (legacy single context ->
    # index 1) or an ordered list of tensors (multi-ref). Each tensor is [B, C, h_i, w_i].
    context_patched_latents = None
    context_patched_latent_image_ids = None
    if context_latents is not None:
        ctx_list = context_latents if isinstance(context_latents, (list, tuple)) else [context_latents]
        ctx_patches, ctx_ids = [], []
        for i, ctx in enumerate(ctx_list):
            ch, cw = ctx.shape[-2], ctx.shape[-1]
            if do_patching == 0:
                ctx_patches.append(pack_latents_no_patch(
                    latents=ctx, batch_size=ctx.shape[0],
                    num_channels_latents=num_channels_latents, height=ch, width=cw,
                ))
                cid = prepare_latent_image_ids(
                    batch_size=ctx.shape[0], height=ch, width=cw, device=device, dtype=noisy_latents.dtype,
                )
            else:
                ctx_patches.append(pack_latents(
                    latents=ctx, batch_size=ctx.shape[0],
                    num_channels_latents=num_channels_latents, height=ch, width=cw,
                ))
                cid = prepare_latent_image_ids(
                    batch_size=ctx.shape[0], height=ch // 2, width=cw // 2, device=device, dtype=noisy_latents.dtype,
                )
            cid[..., 0] = i + 1  # context i -> pos-id i+1
            ctx_ids.append(cid)
        context_patched_latents = torch.cat(ctx_patches, dim=1)            # [B, sum(ctx_seq), C]
        context_patched_latent_image_ids = torch.cat(ctx_ids, dim=0)      # [sum(ctx_seq), 3]

    return {
        "patched_noisy_latents": patched_noisy_latents,
        "patched_latent_image_ids": patched_latent_image_ids,
        "context_patched_latents": context_patched_latents,
        "context_patched_latent_image_ids": context_patched_latent_image_ids,
        "latents_seq_len": patched_noisy_latents.shape[1],
    }


def sample_noise_and_timesteps(
    latents: List[torch.Tensor],
    noise_scheduler,
    do_patching: int,
    vae_scale_factor: int,
    device: torch.device,
    image_dims: List[Tuple[int, int]],
) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Sample noise and timesteps for multi-resolution training.

    Args:
        latents: List of latent tensors, each [C, H_i, W_i]
        image_dims: List of (H, W) pixel dimensions per sample

    Returns:
        (noise_list, sigmas, sigmas_expanded_list, timesteps)
    """
    noise_list = [torch.randn_like(lat) for lat in latents]

    # Sample timesteps per-image based on each image's seq_len
    # (seq_len affects the shift in ShiftedLogitNormalTimestepSampler)
    sigmas_list = []
    for h, w in image_dims:
        if do_patching:
            seq_len = (h // (vae_scale_factor * 2)) * (w // (vae_scale_factor * 2))
        else:
            seq_len = (h // vae_scale_factor) * (w // vae_scale_factor)
        sigma = noise_scheduler.sample(1, seq_len, device=device)
        sigmas_list.append(sigma)

    sigmas = torch.cat(sigmas_list)
    timesteps = sigmas * 1000

    # Expand sigmas per-sample to match each latent's dimensions
    sigmas_expanded_list = []
    for b, sigma_b in enumerate(sigmas):
        while len(sigma_b.shape) < len(latents[b].shape):
            sigma_b = sigma_b.unsqueeze(-1)
        sigmas_expanded_list.append(sigma_b)

    return noise_list, sigmas, sigmas_expanded_list, timesteps


def unpack_model_pred(
    model_pred: torch.Tensor,
    latents_seq_len: int,
    height: int,
    width: int,
    vae_scale_factor: int,
    do_patching: int,
) -> torch.Tensor:
    """Unpack model prediction from sequence to spatial format.

    Used for flash attention path where output is [bsz, seq, D].
    """
    model_pred = model_pred[:, :latents_seq_len]
    if do_patching == 0:
        model_pred = unpack_latents_no_patch(
            latents=model_pred, height=height, width=width, vae_scale_factor=vae_scale_factor
        )
    else:
        model_pred = unpack_latents(
            latents=model_pred, height=height, width=width, vae_scale_factor=vae_scale_factor
        )
    return model_pred


# =============================================================================
# Varlen Utilities
# =============================================================================


def extract_targets_flat(
    output_flat: torch.Tensor,  # [total_img, D] from transformer
    img_lengths: List[int],  # [target+ctx per sample]
    target_lengths: List[int],  # [target per sample]
) -> torch.Tensor:
    """Extract target-only tokens from transformer output.

    The transformer outputs [total_img, D] where each sample has target + context tokens.
    This function extracts only the target tokens (first target_lengths[b] tokens per sample).

    Args:
        output_flat: [total_img, D] flat output from transformer
        img_lengths: Total tokens per sample (target + context)
        target_lengths: Target-only tokens per sample

    Returns:
        [total_target, D] containing only target tokens
    """
    target_outputs = []
    offset = 0
    for total_len, target_len in zip(img_lengths, target_lengths):
        target_outputs.append(output_flat[offset : offset + target_len])
        offset += total_len  # Skip context
    return torch.cat(target_outputs, dim=0)  # [total_target, D]


def compute_loss_flat(
    pred_flat: torch.Tensor,  # [total_target, D]
    target_flat: torch.Tensor,  # [total_target, D]
    bsz: int,
    loss_coeff: float,
) -> torch.Tensor:
    """Compute MSE loss on flat tensors.

    For multi-resolution batches, loss is computed directly on flattened
    tensors without spatial unpacking.

    Uses same loss pattern as standard mode: per-sample mean, then sum over batch,
    then multiply by loss_coeff (which is 1/batch_size for proper averaging).

    Args:
        pred_flat: [total_target, D] predicted values
        target_flat: [total_target, D] ground truth values
        bsz: Batch size (for proper scaling to match standard mode)
        loss_coeff: Loss scaling coefficient

    Returns:
        Scalar loss value
    """
    # Mean over all elements, then scale by bsz to match standard mode's sum-over-batch
    # Standard mode: mean(dim=1).sum() * loss_coeff = bsz * avg * (1/bsz) = avg
    # Flat mode: mean() * bsz * loss_coeff = avg * bsz * (1/bsz) = avg
    per_element_loss = (pred_flat.float() - target_flat.float()) ** 2
    return loss_coeff * bsz * torch.mean(per_element_loss)


def assert_no_context_dropout(drop_rate: float) -> None:
    """Raise if context dropout is configured. Multi-reference contexts do not support it
    (see the multi-ref plan's Global Constraints). Call with each context-affecting drop rate
    (context_drop_rate, both_drop_rate). NOTE: wiring this at startup forbids context dropout
    globally, which conflicts with the dual-CFG capability — gate it before enabling.
    """
    if drop_rate and drop_rate > 0:
        raise RuntimeError(
            "context dropout is not supported with multi-reference contexts; "
            f"set the context-drop rate to 0 (got {drop_rate})"
        )


def apply_context_dropout_varlen(
    context_latents: List[torch.Tensor],
    context_drop_mask: torch.Tensor,
) -> List[Optional[torch.Tensor]]:
    """Apply CFG context dropout for varlen mode.

    Args:
        context_latents: List of context latent tensors
        context_drop_mask: [bsz] bool tensor, True = drop context

    Returns:
        Modified context_latents with dropped samples set to None
    """
    return [None if context_drop_mask[b].item() else ctx for b, ctx in enumerate(context_latents)]


# =============================================================================
# CFG Dropout Utilities
# =============================================================================


@dataclass
class CFGDropoutConfig:
    """Configuration for CFG dropout rates."""

    text_drop_rate: float = 0.1
    context_drop_rate: float = 0.0
    both_drop_rate: float = 0.0


def compute_cfg_dropout_masks(
    cfg_config: CFGDropoutConfig,
    bsz: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute CFG dropout masks for varlen mode.

    Returns:
        Tuple of (text_drop_mask, context_drop_mask), each [bsz] bool tensor
    """
    # Both drop mask
    if cfg_config.both_drop_rate > 0:
        random_p = torch.rand(bsz, device=device, generator=generator)
        both_drop_mask = random_p < cfg_config.both_drop_rate
    else:
        both_drop_mask = torch.zeros(bsz, device=device, dtype=torch.bool)

    # Text dropout mask
    if cfg_config.text_drop_rate > 0:
        random_p = torch.rand(bsz, device=device, generator=generator)
        text_drop_mask = (random_p < cfg_config.text_drop_rate) | both_drop_mask
    else:
        text_drop_mask = both_drop_mask.clone()

    # Context dropout mask
    if cfg_config.context_drop_rate > 0:
        random_p = torch.rand(bsz, device=device, generator=generator)
        context_drop_mask = (random_p < cfg_config.context_drop_rate) | both_drop_mask
    else:
        context_drop_mask = both_drop_mask.clone()

    return text_drop_mask, context_drop_mask


def apply_text_dropout_to_captions(
    captions: List[str],
    text_drop_mask: torch.Tensor,
) -> List[str]:
    """Apply CFG text dropout by replacing dropped captions with empty string.

    This is used for varlen mode where dropout must happen before encoding.
    Empty strings encode to a single BOT token (length=1).
    """
    result = []
    for i, caption in enumerate(captions):
        if text_drop_mask[i].item():
            result.append("")
        else:
            result.append(caption)
    return result


# =============================================================================
# Timing Utilities
# =============================================================================


class DictTimerContext:
    """Context manager for timing that stores results in a dict."""

    def __init__(self, times_dict: Dict[str, float], key: str):
        self.times_dict = times_dict
        self.key = key
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        self.times_dict[self.key] = self.start_event.elapsed_time(self.end_event) / 1000.0


# =============================================================================
# Batch Fetching
# =============================================================================


def fetch_batch(
    iter_: iter,
    get_dataloader: Callable,
    rank: int,
) -> Tuple[tuple, iter, float]:
    """Fetch next batch, reinitializing iterator if exhausted.

    Returns:
        Tuple of (batch, updated_iterator, fetch_time_seconds)
    """
    while True:
        try:
            fetch_start = datetime.now()
            batch = next(iter_)
            fetch_time = (datetime.now() - fetch_start).total_seconds()
            return batch, iter_, fetch_time
        except StopIteration:
            iter_ = iter(get_dataloader())
            logger.info(f"Rank {rank} reinit iterator")
