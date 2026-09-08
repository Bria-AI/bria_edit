"""
Standard training script for FIBO Edit models.

Supports standard diffusion training with CFG dropout.
Supports both flash attention (bsz=1) and varlen attention (bsz>=1) backends.

Usage:
    python train_fibo_edit_standard.py --config path/to/config.yaml
    python train_fibo_edit_standard.py --lora_rank 256 ...
"""

import json
import logging
import os
import random
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import diffusers
import pyrallis
import sagemaker_ssh_helper
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from bria_utils import (
    get_lr_scheduler,
    get_smollm_prompt_embeds_varlen,
    init_text_encoder,
    init_training_scheduler,
    init_wandb,
    pad_embedding,
)
from checkpoint_loader import load_checkpoint
from checkpoint_saver import save_checkpoint
from dataset_factory import (
    DatasetBuilder,
    DatasetConfig,
    DatasetMode,
    calculate_total_batch_size,
)
from diffusers.training_utils import EMAModel
from diffusers.utils import USE_PEFT_BACKEND
from init_handler import TransformerInitResult, init_transformer
from latent_packing import prepare_latent_image_ids
from tqdm.auto import tqdm
from train_common import (
    CFGDropoutConfig,
    DictTimerContext,
    apply_context_dropout_varlen,
    assert_no_context_dropout,
    apply_text_dropout_to_captions,
    auto_discover_data,
    auto_discover_data_interleaved,
    compute_cfg_dropout_masks,
    compute_loss_flat,
    extract_targets_flat,
    fetch_batch,
    get_git_info,
    get_system_info,
    parse_data_config,
    prepare_latents,
    prepare_text_encoder_layers,
    sample_noise_and_timesteps,
    unpack_model_pred,
)
from transformer_bria_repa import Bria4Transformer2DModel
from utils.torch_utils import (
    compile_transformer,
    get_accelerator,
    json_to_data,
)

# Prefer Flash Attention but keep math as fallback for when masks are needed
# Disable mem_efficient to prevent dynamic backend switching which causes timing variation
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable - causes timing variation
torch.backends.cuda.enable_math_sdp(True)  # Keep as fallback for masked attention

# Global variables
TOTAL_BATCH_NO_ACC = None
sagemaker_ssh_helper.setup_and_start_ssh()
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
logger = get_logger(__name__, log_level="INFO")

def _resolve_resume_checkpoint_dir(config: "TrainConfig") -> Optional[str]:
    """Resolve which checkpoint directory load_checkpoint() would have loaded.

    Mirrors CheckpointLoader._resolve_checkpoint_path/_find_latest_checkpoint so the
    sibling per-rank EMA shard can be located after the load completes.
    """
    if config.resume_from_checkpoint == "no":
        return None
    if config.resume_from_checkpoint != "latest":
        path = (
            config.resume_from_checkpoint
            if os.path.isabs(config.resume_from_checkpoint)
            else os.path.join(config.checkpoint_local_path, os.path.basename(config.resume_from_checkpoint))
        )
        return path if os.path.exists(path) else None
    if not os.path.isdir(config.checkpoint_local_path):
        return None
    dirs = [d for d in os.listdir(config.checkpoint_local_path) if d.startswith("checkpoint")]
    if not dirs:
        return None
    dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
    return os.path.join(config.checkpoint_local_path, dirs[-1])


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class TrainConfig:
    """Training configuration for standard mode."""

    # === Core ===
    debug: int = 1
    seed: int = 10

    # === Paths ===
    json_data_input_path: str = "data_input.json.txt"
    checkpoint_local_path: str = "/home/ubuntu/eiga/checkpoints"
    output_dir: str = field(default_factory=lambda: os.environ.get("SM_MODEL_DIR") or "/home/ubuntu/eiga/output")
    s3_bucket_name: str = "hot-ckpt-foundations-useast1"
    s3_prefix: str = "testing"

    # === Model Architecture ===
    transformer_architecture: str = "a6-t"
    lora_rank: int = 0
    lora_init_weights: str = "default"  # "default" (B=zeros, for zero LoRA trick) or "gaussian"
    vae: str = "wan"
    text_encoder_type: str = "smolLM"
    do_patching: int = 0
    num_checkpointing_blocks: int = 0  # 0=disabled, N>0=checkpoint first N blocks

    # === Training ===
    train_batch_size: int = 1
    max_train_steps: int = 10000000
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 10000
    constant_steps: int = -1
    max_grad_norm: float = 0.25

    # === Adam ===
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-15

    # === Data ===
    random_latents_resolution: int = 256
    random_latents: bool = False
    max_sequence_length: int = 3000
    dataloader_num_workers: int = 0
    data_channels: str = "256"
    single_data_dir: Optional[str] = None

    # === DreamBooth (raw images) ===
    use_dreambooth: bool = False
    instance_data_dir: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    image_column: str = "image"
    context_image_column: str = "context_image"
    caption_column: str = "caption"
    raw_caption: bool = False
    vae_model_path: str = "briaai/Fibo-Edit"

    # === Mode 3: Auto-discovery data params ===
    # When set, training script discovers resolutions from mounted directories
    # Format: "name:gpus:batch,name:gpus:batch,..." e.g., "A:32:1" or "TEXT:16:1,IMG:16:2"
    data_config: str = ""
    # Per-channel data mount paths (channel NAME -> path), e.g. {T2I: /data/precomputed/bria4_lt1400}.
    # Preferred over SM_CHANNEL_DATASET_<NAME> env vars: committed in the config and persisted into
    # the saved train_config.yaml, so a run's data paths are recorded and never silently lost/wrong.
    data_paths: Dict[str, str] = field(default_factory=dict)
    # Init/text-encoder mount paths in the config too (config-first, SM_CHANNEL_* env fallback).
    transformer_init_path: str = ""   # pretrained weights to init from (empty -> random / env)
    text_encoder_path: str = ""       # SmolLM dir (empty -> env, then HF default)
    interleave_resolutions: bool = False  # All GPUs load from all resolutions via wds.RandomMix
    # Restrict interleave to one AR family by nominal resolution round(sqrt(W*H)). 0 = unbounded.
    # e.g. 256-only: interleave_max_res=384 ; 1024-only: interleave_min_res=700 (excludes 256 and 512).
    interleave_min_res: int = 0
    interleave_max_res: int = 0

    # === Checkpointing ===
    checkpointing_steps: int = 10000
    resume_from_checkpoint: str = "no"
    reinit_scheduler: int = 0
    reinit_optimizer: int = 0
    use_ema: int = 0
    ema_decay: float = 0.999

    # === Precision & Performance ===
    mixed_precision: str = "bf16"
    allow_tf32: bool = True
    use_fsdp: int = 1
    fsdp_sharding_strategy: str = "hybrid"  # "hybrid" or "full"
    use_torch_compile: int = 0
    regional_compile: int = 0
    compile_dynamic: int = 0  # Use dynamic=True for torch.compile (needed for variable input shapes)
    force_download: bool = True

    # === Logging ===
    wandb_name: str = "fibo_edit_standard"
    logging_dir: str = "logs"

    # === CFG Dropout ===
    text_drop_rate_cfg: float = 0.1
    context_drop_rate_cfg: float = 0.0
    both_drop_rate_cfg: float = 0.0
    edit_instruction_only_prob: float = 0.0

    # === Attention ===
    use_varlen_attention: bool = False  # Use variable-length attention (flash_attn varlen). Required for bsz>1.
    text_pad_length: int = -1  # -1=dynamic (longest in batch), 0=use max_sequence_length, >0=explicit constant
    compile_varlen: int = 0  # 1=compile varlen_attn for additional torch.compile speedup


# =============================================================================
# Validation
# =============================================================================


def validate_config(config: TrainConfig) -> None:
    """Validate config and raise errors for incompatible parameter combinations."""
    # VAE validation
    if config.vae != "wan":
        raise ValueError(f"VAE {config.vae} not supported. Only 'wan' is supported.")

    # Text encoder validation
    if config.text_encoder_type != "smolLM":
        raise ValueError(f"Text encoder {config.text_encoder_type} not supported. Only 'smolLM' is supported.")

    # Context dropout requires varlen attention (flash attention doesn't support attention masking)
    if not config.use_varlen_attention:
        if config.context_drop_rate_cfg > 0 or config.both_drop_rate_cfg > 0:
            raise ValueError(
                "Context dropout (context_drop_rate_cfg, both_drop_rate_cfg) is not supported "
                "with flash attention (use_varlen_attention=False). Only text dropout is supported. "
                "Set use_varlen_attention=True to enable context dropout."
            )

    # Context dropout is not supported with multi-reference contexts (would drop a partial set
    # of the N ordered contexts incoherently). Context drop is unused in practice, so guard it
    # globally rather than per-mode. (If dual-CFG context-drop training is revived, gate this.)
    assert_no_context_dropout(config.context_drop_rate_cfg)
    assert_no_context_dropout(config.both_drop_rate_cfg)

    # Batch size > 1 requires varlen attention (SDPA backend has been deprecated)
    if config.train_batch_size > 1 and not config.use_varlen_attention:
        raise ValueError(
            f"train_batch_size={config.train_batch_size} requires use_varlen_attention=True. "
            "The SDPA attention backend (dense attention matrices) has been deprecated. "
            "Set use_varlen_attention=True for batch sizes > 1."
        )

    # Interleaved multi-resolution loading requires varlen attention
    if config.interleave_resolutions and not config.use_varlen_attention and config.train_batch_size > 1:
        raise ValueError(
            "interleave_resolutions=True with batch_size>1 requires use_varlen_attention=True. "
            "Mixed-resolution batches with batch_size>1 need variable-length attention."
        )

    if config.use_dreambooth:
        if not config.dataset_name and not config.instance_data_dir:
            raise ValueError("DreamBooth mode requires dataset_name or instance_data_dir")
        if config.random_latents:
            raise ValueError("random_latents is incompatible with DreamBooth mode")


# =============================================================================
# Debug Environment Setup
# =============================================================================


def set_debug_env(config: TrainConfig) -> None:
    """Override config values for local debugging."""
    print("In DEBUG mode")

    os.environ["ACCELERATE_MIXED_PRECISION"] = config.mixed_precision
    os.environ["ACCELERATE_DYNAMO_BACKEND"] = "NO"
    os.environ["ACCELERATE_DYNAMO_MODE"] = "default"
    os.environ["ACCELERATE_DYNAMO_USE_FULLGRAPH"] = "False"
    os.environ["ACCELERATE_DYNAMO_USE_DYNAMIC"] = "False"
    os.environ["CLOUD_ENV"] = "AWS"

    # Override training params for fast iteration
    config.resume_from_checkpoint = "no"
    config.checkpointing_steps = 50
    config.gradient_accumulation_steps = 2
    config.lr_warmup_steps = 10
    config.max_train_steps = 50
    config.transformer_architecture = "a1-t"
    config.lora_rank = 128
    config.use_fsdp = 0
    # config.use_torch_compile = 0  # Let CLI override
    # config.random_latents = True  # Disabled for local data testing

    # CFG dropout debug overrides
    config.text_drop_rate_cfg = 0.05
    config.context_drop_rate_cfg = 0.0
    config.both_drop_rate_cfg = 0.0

    print("Were in DEBUG mode")


# =============================================================================
# Model Setup
# =============================================================================


def setup_models(
    config: TrainConfig,
    accelerator: Accelerator,
    transformer_config: Dict[str, Any],
    weight_dtype: torch.dtype,
) -> TransformerInitResult:
    """Initialize transformer for standard training mode."""
    # Convert lora_init_weights config string to the format expected by init_transformer
    lora_init_weights = True if config.lora_init_weights == "default" else config.lora_init_weights

    return init_transformer(
        accelerator=accelerator,
        transformer_config=transformer_config,
        lora_rank=config.lora_rank,
        use_fsdp=config.use_fsdp,
        weight_dtype=weight_dtype,
        gradient_checkpointing=(config.num_checkpointing_blocks > 0),
        debug=config.debug,
        lora_init_weights=lora_init_weights,
        use_varlen_attention=config.use_varlen_attention,
        transformer_init_path=config.transformer_init_path or None,
        # LoRA resume only: load the adapter checkpoint before FSDP wraps the transformer --
        # see init_handler.py's TransformerInitHandler docstring / checkpoint_loader.py's
        # load_lora_weights_for_resume() for why. No-op for lora_rank=0 (full fine-tune) or a
        # fresh run (no checkpoint found yet) -- unchanged behavior in both cases.
        resume_from_checkpoint=config.resume_from_checkpoint,
        checkpoint_local_path=config.checkpoint_local_path,
    )


# =============================================================================
# Dataloader Setup
# =============================================================================


def setup_dataloader(
    config: TrainConfig,
    vae_config: Dict[str, Any],
    rank: int,
) -> Callable:
    """Create dataloader factory for standard training."""
    global TOTAL_BATCH_NO_ACC

    seed = config.seed + rank

    logger.info(f"Hello, Im Rank {rank}, from world size {WORLD_SIZE}")

    if config.use_dreambooth:
        TOTAL_BATCH_NO_ACC = config.train_batch_size * WORLD_SIZE
        logger.info(f"DreamBooth mode. Total batch: {TOTAL_BATCH_NO_ACC * config.gradient_accumulation_steps}")

        dataset_config = DatasetConfig(
            mode=DatasetMode.DREAMBOOTH,
            batch_size=config.train_batch_size,
            edit_instruction_only_prob=config.edit_instruction_only_prob,
            instance_data_dir=config.instance_data_dir,
            dataset_name=config.dataset_name,
            dataset_config_name=config.dataset_config_name,
            cache_dir=config.hf_cache_dir,
            image_column=config.image_column,
            context_image_column=config.context_image_column,
            caption_column=config.caption_column,
            raw_caption=config.raw_caption,
            vae_model=config._vae_model,
            vae_device=config._vae_device,
            vae_weight_dtype=config._vae_weight_dtype,
            num_workers=0,  # Must be 0: collate does VAE encoding on CUDA, incompatible with fork
            prefetch_factor=None,
        )

        builder = DatasetBuilder(config=dataset_config, rank=rank, world_size=WORLD_SIZE, seed=seed)
        return builder.build()

    # Resolve training directories
    if config.random_latents:
        training_dirs = []
        TOTAL_BATCH_NO_ACC = config.train_batch_size * WORLD_SIZE
        logger.info(
            f"Random latents mode enabled. Resolution: {config.random_latents_resolution}, Total batch: {TOTAL_BATCH_NO_ACC}"
        )
    elif config.single_data_dir:
        training_dirs = [config.single_data_dir]
        logger.info(f"Rank {rank}: Using single_data_dir")
    elif WORLD_SIZE > 1:
        # Mode 3: auto-discovery (data_config set)
        # Mode 1 & 2: use json_data_input_path
        if config.data_config and config.interleave_resolutions:
            training_dirs, batch_size, ds_name = auto_discover_data_interleaved(
                rank=rank,
                data_config=config.data_config,
                attach_structured_captions=True,
                min_res=config.interleave_min_res,
                max_res=config.interleave_max_res,
                data_paths=config.data_paths,
            )
            print(f"Rank {rank}: Interleaved data from {len(training_dirs)} resolution dirs (source={ds_name})")
        elif config.data_config:
            training_dirs, _, _, batch_size, _ = auto_discover_data(
                rank=rank,
                data_config=config.data_config,
                attach_structured_captions=True,
            )
            training_dirs = [training_dirs]
            print(f"Rank {rank}: Auto-discovered data from data_config")
        else:
            training_dirs, _, _, _, batch_size = json_to_data(
                rank=rank,
                json_file=config.json_data_input_path,
                attach_structured_captions=True,
            )
            training_dirs = [training_dirs]

        config.train_batch_size = batch_size
        print(f"Rank {rank}: training_dirs={training_dirs}, batch_size={config.train_batch_size}")
    else:
        # Single GPU mode
        if config.data_config and config.interleave_resolutions:
            training_dirs, batch_size, ds_name = auto_discover_data_interleaved(
                rank=rank,
                data_config=config.data_config,
                attach_structured_captions=True,
                min_res=config.interleave_min_res,
                max_res=config.interleave_max_res,
                data_paths=config.data_paths,
            )
            config.train_batch_size = batch_size
            print(f"Rank {rank}: Single-GPU interleaved from {len(training_dirs)} resolution dirs (source={ds_name})")
        elif config.data_config:
            training_dirs, _, _, batch_size, _ = auto_discover_data(
                rank=rank,
                data_config=config.data_config,
                attach_structured_captions=True,
            )
            training_dirs = [training_dirs]
            config.train_batch_size = batch_size
            print(f"Rank {rank}: Single-GPU auto-discovered data")
        else:
            raise ValueError(
                "Single GPU mode requires explicit data configuration. Use one of:\n"
                "  - single_data_dir: Path to training data directory\n"
                "  - json_data_input_path: Path to data config JSON\n"
                "  - data_config: Data config string (e.g., 'A:1:2')\n"
                "  - random_latents: true (for testing without real data)"
            )

    # Calculate total batch size (skip if already set by random_latents)
    if not config.random_latents:
        if WORLD_SIZE == 1:
            # Single GPU mode - just use batch_size directly
            TOTAL_BATCH_NO_ACC = config.train_batch_size
        elif config.data_config:
            # Mode 3: calculate from data_config string
            datasets = parse_data_config(config.data_config)
            TOTAL_BATCH_NO_ACC = sum(ds["batch_size"] * ds["n_gpus"] for ds in datasets)
        else:
            # Mode 1 & 2: use json file
            TOTAL_BATCH_NO_ACC = calculate_total_batch_size(
                json_data_input_path=config.json_data_input_path,
                train_batch_size=config.train_batch_size,
                world_size=WORLD_SIZE,
                single_data_dir=bool(config.single_data_dir),
                debug=config.debug,
                random_latents=config.random_latents,
            )
    print(f"Total batch: {TOTAL_BATCH_NO_ACC * config.gradient_accumulation_steps}")

    # Build dataset configuration
    dataset_config = DatasetConfig(
        mode=DatasetMode.STANDARD,
        batch_size=config.train_batch_size,
        training_dirs=training_dirs,
        random_latents=config.random_latents,
        random_latents_resolution=config.random_latents_resolution,
        edit_instruction_only_prob=config.edit_instruction_only_prob,
        vae_latent_channels=vae_config["latent_channels"],
        vae_compression_rate=vae_config.get("compression_rate", 16),
        interleave_resolutions=config.interleave_resolutions,
        num_workers=config.dataloader_num_workers,
        prefetch_factor=None if config.debug or config.dataloader_num_workers == 0 else 4,
        max_sequence_length=config.max_sequence_length,
    )

    builder = DatasetBuilder(config=dataset_config, rank=rank, world_size=WORLD_SIZE, seed=seed)
    return builder.build()


# =============================================================================
# Batch Unpacking
# =============================================================================


def unpack_batch(batch: tuple) -> Dict[str, Any]:
    """Unpack batch tuple into named dict.

    Args:
        batch: Batch tuple from dataloader (4-tuple for STANDARD mode)
    """
    # STANDARD mode: always list format with image_dims
    pixel_values, context_latents, captions, image_dims = batch
    return {
        "pixel_values": pixel_values,  # List[Tensor], each [C, H_i, W_i]
        "context_latents": context_latents,  # List[Tensor] or None
        "captions": captions,
        "image_dims": image_dims,  # List[(H, W)]
    }


# =============================================================================
# Text Encoding
# =============================================================================


def encode_text(
    captions: List[str],
    get_prompt_embedds: Callable,
    total_num_layers: int,
    null_conditioning_layers: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
    """Encode captions and prepare text encoder layers for flash attention path."""
    encoder_hidden_states, text_encoder_layers, prompt_attention_mask = get_prompt_embedds(captions)
    text_encoder_layers = list(text_encoder_layers)

    text_encoder_layers, null_layers = prepare_text_encoder_layers(
        text_encoder_layers=text_encoder_layers,
        null_conditioning_layers=null_conditioning_layers,
        total_num_layers=total_num_layers,
    )

    encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=torch.float32)
    prompt_attention_mask = prompt_attention_mask.to(device=device, dtype=torch.float32)

    return (
        encoder_hidden_states,
        text_encoder_layers,
        prompt_attention_mask,
        null_layers,
    )


def encode_text_varlen(
    captions: List[str],
    get_prompt_embedds_varlen: Callable,
    total_num_layers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    """Encode captions without padding, return varlen embeddings + text lengths.

    This is used when use_varlen_attention is enabled.
    """
    varlen_embeddings, text_encoder_layers, text_lengths = get_prompt_embedds_varlen(captions)
    text_encoder_layers = list(text_encoder_layers)

    # Prepare layers (trim/extend to match transformer layers)
    text_encoder_layers, _ = prepare_text_encoder_layers(
        text_encoder_layers=text_encoder_layers,
        null_conditioning_layers=None,
        total_num_layers=total_num_layers,
    )

    varlen_embeddings = varlen_embeddings.to(device=device, dtype=torch.float32)
    text_encoder_layers = [layer.to(device=device, dtype=torch.float32) for layer in text_encoder_layers]

    return (
        varlen_embeddings,
        text_encoder_layers,
        text_lengths,
    )


# =============================================================================
# Latent Preparation
# =============================================================================


def prepare_training_latents(
    batch_data: Dict[str, Any],
    vae_config: Dict[str, Any],
    device: torch.device,
    vae_scale_factor: int,
) -> Dict[str, Any]:
    """Extract and scale latents from batch.

    Always returns List[Tensor] format with image_dims.
    """
    shift = vae_config["shift_factor"]
    scale = vae_config["scaling_factor"]

    pixel_values = batch_data["pixel_values"]
    image_dims = batch_data["image_dims"]  # List[(H, W)]

    if not isinstance(pixel_values, list):
        raise ValueError(f"Expected List[Tensor] for pixel_values, got {type(pixel_values)}")

    latents = [(pv.to(device=device, dtype=torch.float32) - shift) * scale for pv in pixel_values]
    bsz = len(pixel_values)

    context = batch_data["context_latents"]
    if context is not None:
        # context is List[List[Tensor]] (ordered contexts per sample), or per-sample None if dropped.
        context = [
            [(c.to(device=device, dtype=torch.float32) - shift) * scale for c in sample_ctx]
            if sample_ctx is not None else None
            for sample_ctx in context
        ]

    return {
        "latents": latents,  # List[Tensor], each [C, H_i, W_i]
        "context_latents": context,  # None | List[None | List[Tensor]] (ordered per sample)
        "bsz": bsz,
        "image_dims": image_dims,  # List[(H, W)]
    }


def create_noisy_latents_and_targets(
    latent_data: Dict[str, Any],
    noise: List[torch.Tensor],
    sigmas_expanded: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Create noisy latents and V-prediction targets.

    Returns:
        (noisy_latents, target) - Lists of tensors
    """
    latents = latent_data["latents"]
    noisy_latents = []
    targets = []
    for b in range(len(latents)):
        noisy_b = sigmas_expanded[b] * noise[b] + (1.0 - sigmas_expanded[b]) * latents[b]
        target_b = noise[b] - latents[b]
        noisy_latents.append(noisy_b)
        targets.append(target_b)
    return noisy_latents, targets


# =============================================================================
# CFG Dropout (Flash Attention Path)
# =============================================================================


def apply_cfg_dropout_flash(
    encoder_hidden_states: torch.Tensor,
    text_encoder_layers: List[torch.Tensor],
    prompt_attention_mask: torch.Tensor,
    null_embedding: torch.Tensor,
    null_conditioning_layers: List[torch.Tensor],
    text_drop_rate: float,
    bsz: int,
    num_text_tokens: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Apply text CFG dropout for flash attention path.

    Note: Context dropout is not supported with flash attention. Use varlen attention
    for context dropout support. Config validation ensures this constraint.
    """
    if text_drop_rate > 0:
        padded_null_embedding, padded_null_attention_mask = pad_embedding(
            prompt_embeds=null_embedding, max_tokens=num_text_tokens
        )
        random_p = torch.rand(bsz, device=device, generator=generator)

        prompt_mask = random_p < text_drop_rate
        prompt_mask = prompt_mask.reshape(bsz, 1, 1)

        encoder_hidden_states = torch.where(prompt_mask, padded_null_embedding, encoder_hidden_states)

        text_encoder_layers = [
            torch.where(
                prompt_mask,
                pad_embedding(
                    prompt_embeds=null_conditioning_layers[i],
                    max_tokens=num_text_tokens,
                )[0],
                text_encoder_layers[i],
            )
            for i in range(len(text_encoder_layers))
        ]

        prompt_mask = prompt_mask.reshape(bsz, 1)
        prompt_attention_mask = torch.where(prompt_mask, padded_null_attention_mask, prompt_attention_mask)

    return (encoder_hidden_states, text_encoder_layers, prompt_attention_mask)


# =============================================================================
# Forward Input Preparation
# =============================================================================


def prepare_forward_inputs_flash(
    encoder_hidden_states: torch.Tensor,
    text_encoder_layers: List[torch.Tensor],
    prompt_attention_mask: torch.Tensor,
    latent_data: Dict[str, Any],
    noisy_latents: List[torch.Tensor],
    target: List[torch.Tensor],
    timesteps: torch.Tensor,
    vae_scale_factor: int,
    device: torch.device,
    null_embedding: torch.Tensor,
    null_layers: List[torch.Tensor],
    text_drop_rate: float,
    generator: torch.Generator,
    do_patching: int = 0,
) -> Dict[str, Any]:
    """Prepare all inputs needed for forward pass (flash attention, bsz=1 only).

    Converts list format from unified collate to batched tensors for flash attention.
    """
    # Convert lists to batched tensors (bsz=1)
    noisy_latents_tensor = noisy_latents[0].unsqueeze(0)  # [1, C, H, W]
    target_tensor = target[0].unsqueeze(0) if target is not None else None

    # context_latents is None (T2I) or List[None | List[Tensor]]; bsz=1 here.
    context_latents = latent_data["context_latents"]
    if context_latents is not None:
        sample_ctx = context_latents[0]  # ordered List[Tensor], or None if dropped
        context_latents = [c.unsqueeze(0) for c in sample_ctx] if sample_ctx else None  # list of [1,C,h_i,w_i]

    # Get pixel dimensions from latent shape
    image_dims = latent_data["image_dims"]
    latent_h, latent_w = image_dims[0]
    height = latent_h * vae_scale_factor
    width = latent_w * vae_scale_factor

    bsz = 1  # Flash attention only supports bsz=1

    # Pack latents
    packed = prepare_latents(
        noisy_latents=noisy_latents_tensor,
        context_latents=context_latents,
        vae_scale_factor=vae_scale_factor,
        height=height,
        width=width,
        device=device,
        do_patching=do_patching,
    )

    # Text IDs (same for all)
    num_text_tokens = encoder_hidden_states.shape[1]
    text_ids = torch.zeros(num_text_tokens, 3).to(device=device, dtype=encoder_hidden_states.dtype)

    # Apply CFG text dropout
    (encoder_hidden_states, text_encoder_layers, prompt_attention_mask) = apply_cfg_dropout_flash(
        encoder_hidden_states=encoder_hidden_states,
        text_encoder_layers=text_encoder_layers,
        prompt_attention_mask=prompt_attention_mask,
        null_embedding=null_embedding,
        null_conditioning_layers=null_layers,
        text_drop_rate=text_drop_rate,
        bsz=bsz,
        num_text_tokens=num_text_tokens,
        generator=generator,
        device=device,
    )

    # Combine latents with context
    patched_noisy = packed["patched_noisy_latents"]
    img_ids = packed["patched_latent_image_ids"]
    latents_seq_len = packed["latents_seq_len"]

    if packed["context_patched_latents"] is not None:
        img_ids = torch.cat([img_ids, packed["context_patched_latent_image_ids"]], dim=0)
        patched_noisy = torch.cat([patched_noisy, packed["context_patched_latents"]], dim=1)

    # Flash attention: no attention mask needed (bsz=1 only, validated in config)
    attention_mask = None

    return {
        "hidden_states": patched_noisy,
        "timesteps": timesteps,
        "encoder_hidden_states": encoder_hidden_states,
        "text_encoder_layers": text_encoder_layers,
        "txt_ids": text_ids,
        "img_ids": img_ids,
        "attention_mask": attention_mask,
        "latents_seq_len": latents_seq_len,
        "target": target_tensor,
        "height": height,
        "width": width,
    }


def pack_sample_tokens_and_ids(noisy_flat, target_img_ids, ctx_list, prepare_ids_fn, device, dtype):
    """Pack one sample's [target | ctx_0 | ctx_1 | ...] tokens and matching img_ids.

    noisy_flat: [target_seq, C] (the target/noisy latent flattened).
    target_img_ids: [target_seq, 3] with channel 0 == 0 (the target marker).
    ctx_list: ordered list of context latents [C, h_i, w_i] (None/empty for no context).
    prepare_ids_fn: (bs, h, w, device, dtype) -> [h*w, 3] position ids.

    Context i (0-based) is stamped with img_ids[..., 0] = i + 1, so pos-ids run target=0,
    contexts=1..N. Returns (tokens [total_seq, C], img_ids [total_seq, 3], ctx_lengths List[int]).
    """
    tokens = [noisy_flat]
    ids = [target_img_ids]
    ctx_lengths = []
    for i, ctx in enumerate(ctx_list or []):
        ch, cw = ctx.shape[-2], ctx.shape[-1]
        ctx_seq = ch * cw
        tokens.append(ctx.permute(1, 2, 0).reshape(ctx_seq, -1))
        cid = prepare_ids_fn(1, ch, cw, device, dtype)
        cid[..., 0] = i + 1
        ids.append(cid)
        ctx_lengths.append(ctx_seq)
    return torch.cat(tokens, dim=0), torch.cat(ids, dim=0), ctx_lengths


def prepare_forward_inputs_varlen(
    varlen_text_embeddings: torch.Tensor,
    text_encoder_layers: List[torch.Tensor],
    text_lengths: List[int],
    latent_data: Dict[str, Any],
    noisy_latents: List[torch.Tensor],
    target: List[torch.Tensor],
    timesteps: torch.Tensor,
    image_dims: List[Tuple[int, int]],
    device: torch.device,
    text_pad_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Prepare forward inputs for varlen mode with BriaTransformer2DModelVarlen.

    The varlen transformer's forward() expects separate flat tensors.
    """
    context_latents = latent_data["context_latents"]
    bsz = latent_data["bsz"]

    # Process per-sample
    target_lengths = []  # H*W per sample (target only)
    context_lengths = []  # ctx H*W per sample (0 if dropped)
    img_lengths = []  # total tokens per sample (target + context)

    all_img_tokens = []
    all_target_for_loss = []
    all_img_ids = []
    all_img_batch_ids = []

    for b in range(bsz):
        latent_h, latent_w = image_dims[b]
        img_seq = latent_h * latent_w
        target_lengths.append(img_seq)

        # Noisy latent for transformer input: [C, H, W] -> [H*W, C]
        noisy_latent = noisy_latents[b]
        noisy_flat = noisy_latent.permute(1, 2, 0).reshape(img_seq, -1)

        # V-prediction target for loss: [C, H, W] -> [H*W, C]
        target_latent = target[b]
        target_flat = target_latent.permute(1, 2, 0).reshape(img_seq, -1)
        all_target_for_loss.append(target_flat)

        # Build image IDs for target
        target_img_ids = prepare_latent_image_ids(
            1, latent_h, latent_w, device, noisy_latent.dtype
        )

        # Pack N ordered contexts (pos-ids 1..N) after the target (pos-id 0). Contexts are
        # EXCLUDED when absent (not zeroed). Refs may differ in size from the target/each other.
        ctx_list = context_latents[b] if context_latents is not None else None
        sample_tokens, sample_img_ids, ctx_lengths = pack_sample_tokens_and_ids(
            noisy_flat, target_img_ids, ctx_list,
            prepare_latent_image_ids, device, noisy_latent.dtype,
        )
        context_lengths.append(sum(ctx_lengths))  # per-sample total context tokens (informational)
        img_lengths.append(target_lengths[-1] + context_lengths[-1])

        all_img_tokens.append(sample_tokens)
        all_img_ids.append(sample_img_ids)
        all_img_batch_ids.append(torch.full((img_lengths[-1],), b, dtype=torch.long, device=device))

    # Concatenate all samples
    hidden_states_flat = torch.cat(all_img_tokens, dim=0)
    target_flat_for_loss = torch.cat(all_target_for_loss, dim=0)
    img_ids_tiled = torch.cat(all_img_ids, dim=0)
    img_batch_ids = torch.cat(all_img_batch_ids, dim=0)

    # Handle constant padding for torch.compile compatibility
    if text_pad_length is not None:
        total_text_tokens = bsz * text_pad_length

        text_batch_ids = []
        for b in range(bsz):
            text_batch_ids.extend([b] * text_pad_length)
        text_batch_ids = torch.tensor(text_batch_ids, dtype=torch.long, device=device)

        txt_ids = torch.zeros(total_text_tokens, 3, device=device, dtype=varlen_text_embeddings.dtype)
        text_lengths_for_interleave = [text_pad_length] * bsz
    else:
        total_text_tokens = sum(text_lengths)

        text_batch_ids = []
        for b, t_len in enumerate(text_lengths):
            text_batch_ids.extend([b] * t_len)
        text_batch_ids = torch.tensor(text_batch_ids, dtype=torch.long, device=device)

        txt_ids = torch.zeros(total_text_tokens, 3, device=device, dtype=varlen_text_embeddings.dtype)
        text_lengths_for_interleave = text_lengths

    return {
        "hidden_states": hidden_states_flat,
        "timesteps": timesteps,
        "encoder_hidden_states": varlen_text_embeddings,
        "txt_ids": txt_ids,
        "img_ids": img_ids_tiled,
        "attention_mask": None,
        "latents_seq_len": None,  # Varlen mode uses img_lengths list instead
        "varlen_params": {
            "text_lengths": text_lengths_for_interleave,
            "actual_text_lengths": text_lengths,
            "img_lengths": img_lengths,
            "target_lengths": target_lengths,
            "context_lengths": context_lengths,
            "image_dims": image_dims,
            "batch_size": bsz,
            "text_batch_ids": text_batch_ids,
            "img_batch_ids": img_batch_ids,
            "text_encoder_layers": text_encoder_layers,
            "target_flat_for_loss": target_flat_for_loss,
        },
    }


# =============================================================================
# Forward Pass
# =============================================================================


def run_forward_pass(
    transformer: nn.Module,
    forward_inputs: Dict[str, Any],
    height: Optional[int],
    width: Optional[int],
    vae_scale_factor: int,
    weight_dtype: torch.dtype,
    use_varlen_attention: bool,
    do_patching: int,
    lora_rank: int,
) -> Dict[str, Any]:
    """Run student forward pass."""
    latents_seq_len = forward_inputs["latents_seq_len"]
    result = {}

    # Joint attention kwargs
    joint_kwargs = {"attention_mask": forward_inputs["attention_mask"]}
    if lora_rank > 0:
        joint_kwargs["scale"] = 1.0

    if use_varlen_attention:
        # Varlen mode
        varlen_params = forward_inputs["varlen_params"]
        varlen_output = transformer(
            text_flat=forward_inputs["encoder_hidden_states"],
            img_flat=forward_inputs["hidden_states"],
            timestep=forward_inputs["timesteps"],
            text_lengths=varlen_params["text_lengths"],
            img_lengths=varlen_params["img_lengths"],
            txt_ids=forward_inputs["txt_ids"],
            img_ids=forward_inputs["img_ids"],
            text_batch_ids=varlen_params["text_batch_ids"],
            img_batch_ids=varlen_params["img_batch_ids"],
            text_encoder_layers=varlen_params["text_encoder_layers"],
            return_dict=False,
            joint_attention_kwargs=joint_kwargs,
            actual_text_lengths=varlen_params.get("actual_text_lengths"),
        )
        model_pred = varlen_output[0]
    else:
        # Flash attention mode
        model_pred, _ = transformer(
            hidden_states=forward_inputs["hidden_states"],
            timestep=forward_inputs["timesteps"],
            encoder_hidden_states=forward_inputs["encoder_hidden_states"],
            text_encoder_layers=forward_inputs["text_encoder_layers"],
            txt_ids=forward_inputs["txt_ids"],
            img_ids=forward_inputs["img_ids"],
            return_dict=False,
            joint_attention_kwargs=joint_kwargs,
        )
        # Unpack to spatial format
        model_pred = unpack_model_pred(
            model_pred=model_pred,
            latents_seq_len=latents_seq_len,
            height=height,
            width=width,
            vae_scale_factor=vae_scale_factor,
            do_patching=do_patching,
        )

    result["model_pred"] = model_pred
    return result


# =============================================================================
# Loss Computation
# =============================================================================


def compute_loss_standard(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    loss_coeff: float,
) -> torch.Tensor:
    """Compute MSE denoising loss for standard mode (flash attention path)."""
    loss = torch.mean(
        ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    ).sum()
    return loss_coeff * loss


# =============================================================================
# Main Training Loop
# =============================================================================


def main():
    config = pyrallis.parse(config_class=TrainConfig)

    # Print CUDA info
    cuda_version = torch.version.cuda
    print(f"PyTorch CUDA Version: {cuda_version}")
    try:
        result_nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        print("CUDA Toolkit version (from nvcc):")
        print(result_nvcc.stdout.split("release ")[1].split(",")[0])
    except Exception:
        print("No nvcc found.")

    # Debug mode setup (before validation to allow overrides)
    if config.debug:
        set_debug_env(config=config)

    # Validate config
    validate_config(config=config)

    # Seeds
    RANK = int(os.environ.get("RANK", 0))
    seed = config.seed + RANK
    set_seed(seed)
    random.seed(seed)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Weight dtype
    weight_dtype = torch.float32
    if config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Accelerator
    accelerator = get_accelerator(config, weight_dtype=weight_dtype)

    # Logger setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f"TORCH_VERSION {torch.__version__}")
    logger.info(f"DIFFUSERS_VERSION {diffusers.__version__}")
    logger.info("Training mode: standard")
    logger.info(f"USE_PEFT_BACKEND: {USE_PEFT_BACKEND}")
    if not USE_PEFT_BACKEND:
        raise RuntimeError("PEFT backend not enabled - LoRA scale parameter will be ignored!")

    # VAE config
    base_dir = Path(__file__).parent.absolute()
    with open(f"{base_dir}/vae_wan.json.out") as f:
        vae_config = json.load(f)
    vae_config["latent_channels"] = 48
    vae_config["shift_factor"] = 0
    vae_config["scaling_factor"] = 1
    vae_config["compression_rate"] = 16

    # Load VAE for DreamBooth mode (on-the-fly encoding)
    if config.use_dreambooth:
        from diffusers import AutoencoderKLWan

        logger.info(f"Loading VAE from {config.vae_model_path}")
        if os.path.isdir(config.vae_model_path):
            vae_model = AutoencoderKLWan.from_pretrained(
                config.vae_model_path,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
        else:
            vae_model = AutoencoderKLWan.from_pretrained(
                config.vae_model_path,
                subfolder="vae",
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )
        vae_model = vae_model.to(accelerator.device, dtype=weight_dtype).eval()
        vae_model.requires_grad_(False)

        # Override vae_config shift/scale with actual VAE values
        vae_config["shift_factor"] = (
            torch.tensor(vae_model.config.latents_mean).reshape(48, 1, 1).to(accelerator.device)
        )
        vae_config["scaling_factor"] = (
            (1 / torch.tensor(vae_model.config.latents_std)).reshape(48, 1, 1).to(accelerator.device)
        )
        logger.info("VAE loaded for DreamBooth mode")

        # Attach to config for passing to DatasetBuilder (runtime-only, not serialized)
        config._vae_model = vae_model
        config._vae_device = accelerator.device
        config._vae_weight_dtype = weight_dtype

    # Transformer config
    if config.debug:
        config_path = os.path.join(base_dir, "bria_transformer_debug.json.out")
    else:
        config_path = os.path.join(base_dir, "bria_transformer.json.out")

    with open(config_path) as f:
        transformer_config = json.load(f)

    # Architecture scaling
    scale_configs = {
        "a1-t": (24, 16, 4, 8, [4, 6, 6]),
        "a2-t": (24, 32, 6, 12, [4, 14, 14]),
        "a3-t": (24, 48, 8, 16, [4, 22, 22]),
        "a4-t": (24, 64, 8, 20, [4, 30, 30]),
        "a5-t": (24, 80, 8, 24, [4, 38, 38]),
        "a6-t": (24, 96, 8, 28, [8, 44, 44]),
        "a7-t": (24, 112, 8, 32, [8, 52, 52]),
        "alpha-t": (24, 128, 8, 38, [16, 56, 56]),
    }

    scale_config = scale_configs[config.transformer_architecture]
    transformer_config["num_attention_heads"] = scale_config[0]
    transformer_config["attention_head_dim"] = scale_config[1]
    transformer_config["num_layers"] = scale_config[2]
    transformer_config["num_single_layers"] = scale_config[3]
    transformer_config["axes_dims_rope"] = scale_config[4]

    # Set in_channels based on VAE and patching
    vae_latent_channels = vae_config["latent_channels"]
    new_in_channels = vae_latent_channels if config.do_patching == 0 else vae_latent_channels * 4
    transformer_config["in_channels"] = new_in_channels

    # Text encoder config
    transformer_config["text_encoder_dim"] = 2048  # smolLM

    # Selective gradient checkpointing
    transformer_config["num_checkpointing_blocks"] = config.num_checkpointing_blocks

    logger.info("------- Transformer Config---------")
    logger.info(transformer_config)
    total_num_layers = transformer_config["num_layers"] + transformer_config["num_single_layers"]

    # Setup models
    models = setup_models(
        config=config,
        accelerator=accelerator,
        transformer_config=transformer_config,
        weight_dtype=weight_dtype,
    )
    transformer = models.transformer
    transformer_unwrapped = models.transformer_unwrapped

    # Text encoder
    tokenizer, text_encoder, get_prompt_embeds_lambda = init_text_encoder(
        text_encoder_type=config.text_encoder_type,
        device=accelerator.device,
        weight_dtype=weight_dtype,
        text_encoder_path=config.text_encoder_path or None,
    )

    # Compute pad_to_length based on config
    if config.text_pad_length == -1:
        text_pad_length = None
    elif config.text_pad_length == 0:
        text_pad_length = config.max_sequence_length
    else:
        text_pad_length = config.text_pad_length

    def get_prompt_embedds(prompts):
        prompt_embeddings, text_encoder_layers, attentions_masks = get_prompt_embeds_lambda(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompts,
            max_sequence_length=config.max_sequence_length,
            pad_to_length=text_pad_length,
        )
        return prompt_embeddings, text_encoder_layers, attentions_masks

    def get_prompt_embedds_varlen(prompts):
        varlen_embeddings, varlen_layers, text_lengths = get_smollm_prompt_embeds_varlen(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=prompts,
            max_sequence_length=config.max_sequence_length,
            pad_to_length=text_pad_length,
        )
        return varlen_embeddings, varlen_layers, text_lengths

    # TF32
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Optimizer
    optimizer_cls = torch.optim.AdamW
    if config.lora_rank > 0:
        parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    else:
        parameters = transformer.parameters()

    optimizer = optimizer_cls(
        parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    # Dataloader
    get_dataloader = setup_dataloader(
        config=config,
        vae_config=vae_config,
        rank=RANK,
    )

    # LR scheduler
    lr_scheduler = get_lr_scheduler(
        name=config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        constant_steps=config.constant_steps * accelerator.num_processes,
    )

    # Accelerator prepare
    if config.use_fsdp:
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    else:
        transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)

    # Set compiled varlen attention (varlen mode only)
    if config.use_varlen_attention:
        transformer_unwrapped.set_compiled_varlen(bool(config.compile_varlen))

    # Torch compile
    if config.use_torch_compile:
        compile_kwargs = {}
        if config.compile_dynamic:
            compile_kwargs["dynamic"] = True
        compile_transformer(
            transformer=transformer,
            transformer_unwrapped=transformer_unwrapped,
            regional=config.regional_compile,
            **compile_kwargs,
        )

    # EMA model. Created after accelerator.prepare so the shadow params match the
    # (possibly sharded) parameter shapes used in training. We do NOT register it with
    # accelerator.register_for_checkpointing because that path writes a single
    # custom_checkpoint_0.pkl from the global main process only -- under FSDP every rank
    # would then reload rank 0's local shadow_params shard. We save/load per-rank shards
    # explicitly below instead.
    ema_transformer = None
    if config.use_ema:
        logger.info(f"Initializing EMA with decay {config.ema_decay}")
        ema_transformer = EMAModel(
            transformer.parameters(),
            decay=config.ema_decay,
            model_cls=Bria4Transformer2DModel,
            model_config=transformer_unwrapped.config,
        )
        ema_transformer.to(accelerator.device)

    # Wandb
    if accelerator.is_main_process and not config.debug:
        init_wandb(
            accelerator=accelerator,
            args=config,
            project_name=os.environ.get("WANDB_PROJECT", "fibo-edit-standard"),
            extend_name=False,
        )

        # Log reproducibility info to wandb
        wandb.config.update(get_git_info(), allow_val_change=True)
        wandb.config.update(get_system_info(), allow_val_change=True)

        # Save config as YAML artifact
        os.makedirs(config.output_dir, exist_ok=True)
        config_path = os.path.join(config.output_dir, "train_config.yaml")
        with open(config_path, "w") as f:
            pyrallis.dump(config, f)
        config_artifact = wandb.Artifact(f"config-{wandb.run.id}", type="config")
        config_artifact.add_file(config_path)
        wandb.log_artifact(config_artifact)
        logger.info(f"Saved config artifact to wandb: {config_path}")

    logger.info("***** Running training *****")
    logger.info(f"diffusers version: {diffusers.__version__}")
    logger.info(f"Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {config.max_train_steps}")

    global_step = int(os.environ.get("GLOBAL_STEP", 0))

    # Resume from checkpoint
    if config.resume_from_checkpoint != "no":
        global_step = load_checkpoint(
            accelerator=accelerator,
            checkpoint_dir=config.checkpoint_local_path,
            resume_from=config.resume_from_checkpoint,
            is_lora=config.lora_rank > 0,
            transformer=transformer_unwrapped,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            reinit_optimizer=config.reinit_optimizer,
            reinit_scheduler=config.reinit_scheduler,
            # LoRA resume already loaded the adapter weights pre-FSDP-wrap, in setup_models()
            # (see that call's own comment) -- don't try (and fail) to reload them here.
            lora_weights_already_loaded=models.lora_weights_loaded_early,
        )

        if config.use_ema and ema_transformer is not None:
            ckpt_dir = _resolve_resume_checkpoint_dir(config)
            if ckpt_dir is not None:
                ema_rank_path = os.path.join(ckpt_dir, f"ema_state_rank_{accelerator.process_index}.pt")
                if not os.path.isfile(ema_rank_path):
                    raise FileNotFoundError(
                        f"Per-rank EMA shard not found at {ema_rank_path}. Resuming with "
                        "use_ema=1 requires checkpoints saved with EMA enabled."
                    )
                logger.info(f"Loading per-rank EMA shard from {ema_rank_path}")
                rank_ema_state = torch.load(ema_rank_path, map_location="cpu", weights_only=False)
                ema_transformer.load_state_dict(rank_ema_state)
                ema_transformer.to(accelerator.device)

    # Reinit optimizer/scheduler if requested
    if config.reinit_optimizer:
        del optimizer
        optimizer = optimizer_cls(
            parameters,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )
        accelerator._optimizers = []
        optimizer = accelerator.prepare_optimizer(optimizer)
        logger.info("Replacing Optimizer")

    if config.reinit_optimizer or config.reinit_scheduler:
        if not config.reinit_optimizer:
            # optimizer wasn't rebuilt above, so its param_groups still carry the checkpoint's
            # old lr AND old initial_lr (set by the checkpoint's own scheduler at its
            # construction time). LRScheduler.__init__ does group.setdefault("initial_lr", ...),
            # a no-op when the key already exists -- so base_lrs would silently keep the stale
            # initial_lr (and the immediate _initial_step() call would overwrite lr right back to
            # it) unless initial_lr is cleared here too, alongside lr.
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate
                group.pop("initial_lr", None)
        del lr_scheduler
        # The reinit'd scheduler starts fresh at internal step 0 (it is NOT fast-forwarded to
        # global_step), and the training loop only steps it (max_train_steps - global_step) times.
        # So span it over the REMAINING steps, not the absolute max_train_steps — otherwise a
        # decay set to e.g. 60k only advances ~18k and never reaches 0. For a fresh run
        # (global_step=0) remaining == max_train_steps, so non-resume behavior is unchanged.
        remaining_steps = config.max_train_steps - global_step
        lr_scheduler = get_lr_scheduler(
            name=config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=remaining_steps * accelerator.num_processes,
            constant_steps=config.constant_steps * accelerator.num_processes,
        )
        accelerator._schedulers = []
        lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)
        logger.info(f"Replacing LR Scheduler ({config.lr_scheduler}, span={remaining_steps} remaining steps)")

    logger.info(f"Using adam with lr: {config.learning_rate}, beta2: {config.adam_beta2}")

    # Progress bar
    progress_bar = tqdm(
        range(global_step, config.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # Noise scheduler
    noise_scheduler = init_training_scheduler()

    # Pre-compute null embeddings (for flash attention CFG dropout)
    null_conditioning, null_conditioning_layers, _ = get_prompt_embedds([""])
    logger.info("Using empty prompt for null embeddings")
    assert null_conditioning.shape[0] == 1
    null_conditioning = null_conditioning.repeat(config.train_batch_size, 1, 1).to(dtype=torch.float32)
    null_conditioning_layers = [
        layer.repeat(config.train_batch_size, 1, 1).to(dtype=torch.float32) for layer in null_conditioning_layers
    ]

    vae_scale_factor = (
        vae_config["compression_rate"]
        if "compression_rate" in vae_config
        else 2 ** (len(vae_config["block_out_channels"]) - 1)
    )

    # Set model to train mode
    transformer.train()

    # Training state
    train_loss = 0.0
    now = datetime.now()
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    # CFG dropout config
    cfg_dropout_config = CFGDropoutConfig(
        text_drop_rate=config.text_drop_rate_cfg,
        context_drop_rate=config.context_drop_rate_cfg,
        both_drop_rate=config.both_drop_rate_cfg,
    )

    iter_ = iter(get_dataloader())
    for step in range(
        global_step * config.gradient_accumulation_steps,
        config.max_train_steps * config.gradient_accumulation_steps,
    ):
        batch, iter_, fetch_time = fetch_batch(
            iter_=iter_,
            get_dataloader=get_dataloader,
            rank=RANK,
        )
        batch_data = unpack_batch(batch=batch)
        captions = batch_data["captions"]
        step_times = {}

        with accelerator.accumulate(transformer):
            # STEP 1: Encode Text
            with DictTimerContext(times_dict=step_times, key="text_encode"):
                if config.use_varlen_attention:
                    # Varlen mode: compute CFG dropout BEFORE encoding
                    bsz = config.train_batch_size
                    text_drop_mask, context_drop_mask = compute_cfg_dropout_masks(
                        cfg_config=cfg_dropout_config,
                        bsz=bsz,
                        generator=generator,
                        device=accelerator.device,
                    )

                    # Apply CFG dropouts
                    captions_for_encoding = apply_text_dropout_to_captions(
                        captions=captions,
                        text_drop_mask=text_drop_mask,
                    )
                    if batch_data.get("context_latents") is not None:
                        batch_data["context_latents"] = apply_context_dropout_varlen(
                            context_latents=batch_data["context_latents"],
                            context_drop_mask=context_drop_mask,
                        )

                    # Encode with dropout already applied
                    enc_hidden, text_layers, text_lengths = encode_text_varlen(
                        captions=captions_for_encoding,
                        get_prompt_embedds_varlen=get_prompt_embedds_varlen,
                        total_num_layers=total_num_layers,
                        device=accelerator.device,
                    )
                else:
                    # Flash attention mode: padded [bsz, max_len, dim]
                    enc_hidden, text_layers, prompt_mask, null_layers = encode_text(
                        captions=captions,
                        get_prompt_embedds=get_prompt_embedds,
                        total_num_layers=total_num_layers,
                        null_conditioning_layers=null_conditioning_layers,
                        device=accelerator.device,
                    )

            # STEP 2: Prepare Latents
            with DictTimerContext(times_dict=step_times, key="latent_prep"):
                latent_data = prepare_training_latents(
                    batch_data=batch_data,
                    vae_config=vae_config,
                    device=accelerator.device,
                    vae_scale_factor=vae_scale_factor,
                )

            # STEP 3: Sample Noise & Timesteps
            with DictTimerContext(times_dict=step_times, key="noise_sample"):
                image_dims = latent_data["image_dims"]
                noise, sigmas, sigmas_exp, timesteps = sample_noise_and_timesteps(
                    latents=latent_data["latents"],
                    noise_scheduler=noise_scheduler,
                    do_patching=config.do_patching,
                    vae_scale_factor=vae_scale_factor,
                    device=accelerator.device,
                    image_dims=image_dims,
                )

            # STEP 4: Create Noisy Latents & Targets
            noisy_latents, target = create_noisy_latents_and_targets(
                latent_data=latent_data,
                noise=noise,
                sigmas_expanded=sigmas_exp,
            )

            # STEP 5: Prepare Forward Inputs
            with DictTimerContext(times_dict=step_times, key="forward_prep"):
                if config.use_varlen_attention:
                    latent_device = noisy_latents[0].device
                    forward_inputs = prepare_forward_inputs_varlen(
                        varlen_text_embeddings=enc_hidden,
                        text_encoder_layers=text_layers,
                        text_lengths=text_lengths,
                        latent_data=latent_data,
                        noisy_latents=noisy_latents,
                        target=target,
                        timesteps=timesteps,
                        image_dims=image_dims,
                        device=latent_device,
                        text_pad_length=text_pad_length,
                    )
                else:
                    forward_inputs = prepare_forward_inputs_flash(
                        encoder_hidden_states=enc_hidden,
                        text_encoder_layers=text_layers,
                        prompt_attention_mask=prompt_mask,
                        latent_data=latent_data,
                        noisy_latents=noisy_latents,
                        target=target,
                        timesteps=timesteps,
                        vae_scale_factor=vae_scale_factor,
                        device=accelerator.device,
                        null_embedding=null_conditioning,
                        null_layers=null_layers,
                        text_drop_rate=config.text_drop_rate_cfg,
                        generator=generator,
                        do_patching=config.do_patching,
                    )

            # STEP 6: Forward Pass
            with DictTimerContext(times_dict=step_times, key="forward"):
                fwd_h = forward_inputs.get("height")
                fwd_w = forward_inputs.get("width")
                outputs = run_forward_pass(
                    transformer=transformer,
                    forward_inputs=forward_inputs,
                    height=fwd_h,
                    width=fwd_w,
                    vae_scale_factor=vae_scale_factor,
                    weight_dtype=weight_dtype,
                    use_varlen_attention=config.use_varlen_attention,
                    do_patching=config.do_patching,
                    lora_rank=config.lora_rank,
                )

            # STEP 7: Compute Loss
            loss_coeff = WORLD_SIZE / TOTAL_BATCH_NO_ACC

            if config.use_varlen_attention:
                varlen_params = forward_inputs["varlen_params"]
                pred_flat = extract_targets_flat(
                    outputs["model_pred"],
                    varlen_params["img_lengths"],
                    varlen_params["target_lengths"],
                )
                target_flat = varlen_params["target_flat_for_loss"]
                bsz = varlen_params["batch_size"]
                loss = compute_loss_flat(pred_flat, target_flat, bsz, loss_coeff)
            else:
                flash_target = forward_inputs["target"]
                loss = compute_loss_standard(
                    model_pred=outputs["model_pred"],
                    target=flash_target,
                    loss_coeff=loss_coeff,
                )

            # Gather losses
            train_loss += accelerator.gather(loss.detach()).mean().item() / config.gradient_accumulation_steps

            # STEP 8: Backward
            with DictTimerContext(times_dict=step_times, key="backward"):
                accelerator.backward(loss)

            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), config.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Logging step
        if accelerator.sync_gradients:
            if config.use_ema:
                ema_transformer.step(transformer.parameters())

            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                logger.info(f"train_loss: {train_loss}")
                after = datetime.now() - now
                now = datetime.now()

                log_params = {
                    # Metrics
                    "metrics/train_loss": train_loss,
                    "metrics/learning_rate": optimizer.param_groups[0]["lr"],
                    "metrics/grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                    # Timings
                    "timings/batch_time_sec": after.total_seconds(),
                    "timings/fetch_batch": fetch_time,
                }
                if config.use_ema:
                    log_params["metrics/ema_decay"] = ema_transformer.cur_decay_value
                # Add step_times with timings/ prefix
                for key, value in step_times.items():
                    log_params[f"timings/{key}"] = value

                # Add torch.compile recompilation tracking
                if config.use_torch_compile:
                    dynamo_counters = torch._dynamo.utils.counters["stats"]
                    log_params["compile/unique_graphs"] = dynamo_counters.get("unique_graphs", 0)
                    log_params["compile/graph_breaks"] = dynamo_counters.get("graph_breaks", 0)

                # Add text_lengths stats for varlen mode
                if config.use_varlen_attention and text_lengths is not None:
                    log_params["varlen/text_lengths_min"] = min(text_lengths)
                    log_params["varlen/text_lengths_max"] = max(text_lengths)
                    log_params["varlen/text_lengths_unique"] = len(set(text_lengths))
                    log_params["varlen/text_lengths_total"] = sum(text_lengths)
                    if text_pad_length is not None:
                        log_params["varlen/text_pad_length"] = text_pad_length
                        log_params["varlen/total_tokens_constant"] = len(text_lengths) * text_pad_length

                # Sequence length logging (varlen mode)
                if config.use_varlen_attention and "varlen_params" in forward_inputs:
                    vp = forward_inputs["varlen_params"]
                    log_params["seq/img_lengths_min"] = min(vp["img_lengths"])
                    log_params["seq/img_lengths_max"] = max(vp["img_lengths"])
                    log_params["seq/img_lengths_sum"] = sum(vp["img_lengths"])
                    log_params["seq/total_tokens"] = sum(vp["img_lengths"]) + sum(vp["text_lengths"])
                    if "image_dims" in vp:
                        dims = vp["image_dims"]
                        log_params["seq/resolutions"] = str([(w, h) for w, h in dims])

                # Sequence length logging (flash mode)
                if not config.use_varlen_attention and "latents_seq_len" in forward_inputs:
                    seq_len = forward_inputs["latents_seq_len"]
                    text_len = (
                        forward_inputs["encoder_hidden_states"].shape[1]
                        if "encoder_hidden_states" in forward_inputs
                        else 0
                    )
                    log_params["seq/img_seq_len"] = seq_len
                    log_params["seq/text_seq_len"] = text_len
                    log_params["seq/total_tokens"] = seq_len + text_len

                # Print seq metrics in debug mode (wandb is disabled)
                if config.debug:
                    seq_metrics = {k: v for k, v in log_params.items() if k.startswith("seq/")}
                    if seq_metrics:
                        logger.info(f"seq_metrics: {seq_metrics}")

                if not config.debug:
                    wandb.log(log_params, step=global_step)

            train_loss = 0.0

            # Checkpointing
            if (global_step - 1) % config.checkpointing_steps == 0 and (global_step - 1) > 0:
                if config.use_ema:
                    ema_save_path = os.path.join(config.checkpoint_local_path, f"checkpoint_{global_step - 1:06d}")

                    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
                        accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
                    accelerator.wait_for_everyone()

                    ema_final_path = os.path.join(ema_save_path, "transformer_ema.bin")
                    logger.info(f"Saving EMA state dict to {ema_final_path}")

                    # Gather full EMA params via the transformer. Collective op - must run on
                    # every rank; only rank 0 receives the populated state_dict.
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())
                    ema_state_dict = accelerator.get_state_dict(transformer, unwrap=False)
                    # accelerator.get_state_dict() falls back to plain model.state_dict() outside
                    # FSDP, whose tensors are .detach()'d VIEWS sharing storage with the live
                    # params -- restore() below would silently overwrite them back to the live
                    # (non-EMA) values before we ever save them. Clone to break the aliasing.
                    ema_state_dict = {k: v.clone() for k, v in ema_state_dict.items()}
                    ema_transformer.restore(transformer.parameters())

                    if accelerator.is_main_process:
                        ema_state_dict = {
                            k.replace("module.", "").replace("._orig_mod", ""): v
                            for k, v in ema_state_dict.items()
                        }
                        os.makedirs(ema_save_path, exist_ok=True)
                        torch.save(ema_state_dict, ema_final_path)
                        del ema_state_dict
                    accelerator.wait_for_everyone()

                    # Per-rank EMA shard for resume. Each rank's shadow_params hold its own
                    # FSDP-local slice, so we write one file per rank (sized like a single
                    # shard). On resume each rank reads its own shard back. transformer_ema.bin
                    # above is the full-gathered copy used by eval/inference.
                    os.makedirs(ema_save_path, exist_ok=True)
                    ema_rank_path = os.path.join(ema_save_path, f"ema_state_rank_{accelerator.process_index}.pt")
                    torch.save(ema_transformer.state_dict(), ema_rank_path)
                    accelerator.wait_for_everyone()

                save_checkpoint(
                    accelerator=accelerator,
                    step=global_step - 1,
                    checkpoint_dir=config.checkpoint_local_path,
                    transformer=transformer,
                    is_lora=config.lora_rank > 0,
                    transformer_unwrapped=transformer_unwrapped,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    data_config_path=config.json_data_input_path,
                )
                now = datetime.now()

        logs = {"step_loss": loss.detach().item(), "global_step": global_step}
        progress_bar.set_postfix(**logs)

        if global_step >= config.max_train_steps:
            break

    logger.info("Waiting for everyone :)")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["FI_PROVIDER"] = "efa"
    os.environ["FI_EFA_USE_DEVICE_RDMA"] = "1"
    os.environ["NCCL_MIN_NCHANNELS"] = "8"
    os.environ["NCCL_NET_GDR_LEVEL"] = "PHB"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_CROSS_NIC"] = "1"

    main()
