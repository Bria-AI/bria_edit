"""
Transformer initialization utilities for FIBO Edit training.

Trimmed to what train_fibo_edit_standard.py actually uses: this handler
originally also built the teacher transformer (CFG distillation) and the
reference transformer (DPO) when create_teacher/create_ref were passed True.
train_fibo_edit_standard.py's setup_models() always passes both False (those
are separate training scripts' job), so that machinery (_setup_teacher_model,
_setup_ref_model, and _load_lora_weights, which only the teacher path called)
was unreachable dead code here -- removed. See README.md in this directory.
"""

import os
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers.training_utils import cast_training_params
from safetensors.torch import load_file

from utils.common import get_env_prefix
from model.lora import add_lora, load_lora
from model.transformer import Bria4Transformer2DModel

logger = logging.getLogger(__name__)


@dataclass
class TransformerInitResult:
    """Result of transformer initialization."""
    transformer: nn.Module  # Main trainable transformer (with LoRA if enabled)
    transformer_unwrapped: nn.Module  # Reference to unwrapped model for save_pretrained
    is_lora: bool
    # True when a LoRA resume's adapter weights were already loaded here, before FSDP-wrap
    # (see TransformerInitHandler's resume_from_checkpoint/checkpoint_local_path args). The
    # caller's own later load_checkpoint() call should then pass lora_weights_already_loaded=True
    # so it doesn't try (and fail) to reload them post-FSDP-wrap. False for every existing
    # caller that doesn't pass resume_from_checkpoint/checkpoint_local_path -- unchanged behavior.
    lora_weights_loaded_early: bool = False


class TransformerInitHandler:
    """
    Transformer initialization for standard (non-distillation, non-DPO) training.

    Handles:
    - Meta device pattern for memory-efficient loading
    - TRANSFORMER_INIT env var for pretrained weights
    - LORA_INIT env var for pretrained LoRA weights
    - LoRA setup via add_lora() or load_lora()
    - FSDP preparation

    Example:
        handler = TransformerInitHandler(
            accelerator=accelerator,
            transformer_config=transformer_config,
            lora_rank=args.lora_rank,
            use_fsdp=args.use_fsdp,
            weight_dtype=weight_dtype,
        )
        result = handler.initialize()
        transformer = result.transformer
    """

    def __init__(
        self,
        accelerator: Accelerator,
        transformer_config: Dict[str, Any],
        lora_rank: int = 0,
        use_fsdp: bool = False,
        weight_dtype: torch.dtype = torch.float32,
        gradient_checkpointing: bool = False,
        debug: bool = False,
        lora_init_weights: Union[bool, str] = True,
        use_varlen_attention: bool = False,
        parallel_checkpoint_load: bool = True,
        transformer_init_path: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        checkpoint_local_path: Optional[str] = None,
    ):
        """
        Initialize the transformer init handler.

        Args:
            accelerator: HuggingFace Accelerator instance
            transformer_config: Transformer configuration dict
            lora_rank: LoRA rank (0 for no LoRA)
            use_fsdp: Whether to use FSDP
            weight_dtype: Weight dtype for model (e.g., torch.bfloat16)
            gradient_checkpointing: Enable gradient checkpointing
            debug: Debug mode (skip weight loading)
            lora_init_weights: LoRA initialization method
                - True (default): Standard init, A=Kaiming, B=zeros
                - "gaussian": Both A and B are Gaussian random
            use_varlen_attention: Convert to varlen attention transformer variant
            parallel_checkpoint_load: Enable parallel checkpoint loading across all FSDP ranks.
                When True (default), all ranks load weights simultaneously via mmap, which is
                faster than rank-0-only loading. Set to False to revert to sequential loading.
            resume_from_checkpoint: Same value as TrainConfig.resume_from_checkpoint. Optional;
                when set (and not "no") together with checkpoint_local_path, a LoRA run's
                (lora_rank > 0) adapter weights are loaded here in initialize(), BEFORE FSDP
                wraps the transformer -- see the FSDP preparation section below and
                checkpoint_loader.py's load_lora_weights_for_resume() docstring for why this
                ordering matters. None/"no" (the default) is a full no-op, identical to every
                existing caller that doesn't pass this.
            checkpoint_local_path: Same value as TrainConfig.checkpoint_local_path -- the
                directory load_lora_weights_for_resume() searches for the checkpoint to resume.
        """
        self.accelerator = accelerator
        self.transformer_config = transformer_config
        self.lora_rank = lora_rank
        self.use_fsdp = use_fsdp
        self.weight_dtype = weight_dtype
        self.gradient_checkpointing = gradient_checkpointing
        self.debug = debug
        self.lora_init_weights = lora_init_weights
        self.use_varlen_attention = use_varlen_attention
        self.parallel_checkpoint_load = parallel_checkpoint_load
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_local_path = checkpoint_local_path

        # Get env vars
        # config value first (committed + persisted), then SM_CHANNEL_TRANSFORMER_INIT env fallback
        self.transformer_init_path = transformer_init_path or os.environ.get(f"{get_env_prefix()}_TRANSFORMER_INIT")
        self.lora_init_path = os.environ.get(f"{get_env_prefix()}_LORA_INIT")

    def initialize(self) -> TransformerInitResult:
        """
        Initialize the transformer(s).

        Returns:
            TransformerInitResult with initialized models
        """
        # Create and load main transformer
        transformer, _state_dict = self._create_and_load_transformer()

        # Convert to varlen attention transformer if requested (must happen before LoRA and FSDP)
        if self.use_varlen_attention:
            from model.transformer_varlen import BriaTransformer2DModelVarlen
            transformer = BriaTransformer2DModelVarlen.from_standard_model(transformer)
            logger.info("Converted transformer to BriaTransformer2DModelVarlen for varlen attention")

            # Freeze repa_mlp if not using REPA — varlen conversion creates fresh params
            # with requires_grad=True, but these are unused when flag_repa=False
            if hasattr(transformer, "flag_repa") and not transformer.flag_repa:
                transformer.repa_mlp.requires_grad_(False)
                logger.info("Froze repa_mlp (flag_repa=False)")

        # Setup LoRA if needed (after varlen conversion so PeftModel wraps the varlen model)
        is_lora = self.lora_rank > 0
        transformer = self._setup_lora(transformer)

        # Resume a LoRA run's adapter weights HERE, before FSDP wraps/shards the transformer
        # below -- see load_lora_weights_for_resume()'s docstring (checkpoint_loader.py) for
        # why loading LoRA weights AFTER FSDP-wrap is unsafe for small target layers. No-op
        # (and lora_weights_loaded_early stays False) unless both resume_from_checkpoint and
        # checkpoint_local_path are set and a checkpoint is actually found -- every existing
        # caller that doesn't pass them is unaffected.
        lora_weights_loaded_early = False
        if is_lora and self.resume_from_checkpoint not in (None, "no") and self.checkpoint_local_path:
            from checkpoint.loader import CheckpointLoader
            resume_loader = CheckpointLoader(
                accelerator=self.accelerator,
                checkpoint_dir=self.checkpoint_local_path,
                is_lora=True,
            )
            lora_weights_loaded_early, resumed_from = resume_loader.load_lora_weights_for_resume(
                self.resume_from_checkpoint, transformer
            )
            if lora_weights_loaded_early:
                logger.info(f"Loaded LoRA adapter weights early (pre-FSDP) from {resumed_from}")
                # The checkpoint's saved adapter dtype (commonly float32) can differ from the
                # bf16 that _setup_lora()'s own cast_training_params() already applied to the
                # freshly-added (pre-load) params -- FSDP's flatten requires every parameter in
                # a unit to share one dtype, so re-cast after overwriting values with the loaded
                # checkpoint. Confirmed this session: without this, FSDP prepare() below fails
                # with "Must flatten tensors with uniform dtype but got torch.bfloat16 and
                # torch.float32". Same cast_training_params call/dtype _setup_lora() already uses.
                cast_training_params([transformer], dtype=self.weight_dtype if self.use_fsdp else torch.float32)

        # Log parameter counts
        self._log_param_counts(transformer)

        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()

        # Keep unwrapped reference for save_pretrained
        transformer_unwrapped = transformer

        # FSDP preparation
        if self.use_fsdp:
            # Sync all ranks before FSDP prepare
            self.accelerator.wait_for_everyone()
            transformer = self.accelerator.prepare(transformer)
            # Sync after prepare to ensure all ranks complete FSDP setup
            self.accelerator.wait_for_everyone()

        return TransformerInitResult(
            transformer=transformer,
            transformer_unwrapped=transformer_unwrapped,
            is_lora=is_lora,
            lora_weights_loaded_early=lora_weights_loaded_early,
        )

    def _create_and_load_transformer(self) -> tuple:
        """Create transformer and load pretrained weights."""
        # Determine if we should use meta device
        use_meta_device = self.transformer_init_path is not None and not self.debug

        # For varlen attention, don't use meta device (from_standard_model needs real tensors)
        if self.use_varlen_attention:
            use_meta_device = False

        # Create transformer
        with torch.device("meta") if use_meta_device else nullcontext():
            transformer = Bria4Transformer2DModel.from_config(self.transformer_config)

        state_dict = None

        # Load pretrained weights
        if self.transformer_init_path and not self.debug:
            state_dict = self._load_transformer_weights(transformer)
        elif self.debug:
            # Debug mode: get state_dict from randomly initialized model
            state_dict = transformer.state_dict()

        return transformer, state_dict

    def _load_transformer_weights(self, transformer: nn.Module) -> Optional[Dict]:
        """Load pretrained transformer weights from various formats."""
        # Parallel loading: all ranks load checkpoint simultaneously when enabled (default)
        # This is faster than rank-0-only because:
        # - mmap=True shares OS page cache across processes
        # - safetensors supports parallel zero-copy reads
        # - FSDP's sync_module_states becomes no-op when weights are identical
        if self.use_fsdp and not self.parallel_checkpoint_load:
            if not self.accelerator.is_main_process:
                return None

        transformer_init = self.transformer_init_path

        # Find weight file
        if os.path.exists(f"{transformer_init}/pytorch_model.bin"):
            weight_path = f"{transformer_init}/pytorch_model.bin"
        elif os.path.exists(f"{transformer_init}/pytorch_model_fsdp.bin"):
            weight_path = f"{transformer_init}/pytorch_model_fsdp.bin"
        elif os.path.exists(f"{transformer_init}/model.safetensors"):
            weight_path = f"{transformer_init}/model.safetensors"
        else:
            raise FileNotFoundError(f"No weights found at {transformer_init}")

        logger.info(f"\n--------Loading transformer weights from {weight_path}--------\n")

        from time import perf_counter
        use_mmap = self.use_fsdp  # mmap only for FSDP (shared page cache across ranks)
        load_device = str(self.accelerator.device) if not self.use_fsdp else "cpu"
        load_dtype = self.weight_dtype if self.lora_rank > 0 else None

        t0 = perf_counter()
        if weight_path.endswith('.safetensors'):
            state_dict = load_file(weight_path, device="cpu")
        else:
            state_dict = torch.load(weight_path, map_location="cpu", mmap=use_mmap)
        logger.info(f"torch.load took {perf_counter()-t0:.1f}s (mmap={use_mmap})")

        # Convert dtype + move to GPU in one pass per tensor (avoids full fp32 copy on GPU)
        if load_dtype is not None:
            t1 = perf_counter()
            state_dict = {k: v.to(device=load_device, dtype=load_dtype) for k, v in state_dict.items()}
            logger.info(f"state_dict to {load_dtype} on {load_device} took {perf_counter()-t1:.1f}s")
        elif load_device != "cpu":
            t1 = perf_counter()
            state_dict = {k: v.to(device=load_device) for k, v in state_dict.items()}
            logger.info(f"state_dict to {load_device} took {perf_counter()-t1:.1f}s")

        t2 = perf_counter()
        transformer.load_state_dict(state_dict, assign=True)
        logger.info(f"load_state_dict took {perf_counter()-t2:.1f}s")

        return state_dict

    def _setup_lora(self, transformer: nn.Module) -> nn.Module:
        """Setup LoRA on the transformer."""
        logger.info(f"Using precision of {self.weight_dtype}")

        if self.lora_rank > 0:
            from time import perf_counter
            logger.info(f"Using LoRA with rank {self.lora_rank}")

            t0 = perf_counter()
            transformer.requires_grad_(False)
            logger.info(f"requires_grad_(False) took {perf_counter()-t0:.1f}s")

            t0 = perf_counter()
            model_dtype = getattr(transformer, "dtype", None)
            if model_dtype is None or model_dtype != self.weight_dtype:
                transformer.to(dtype=self.weight_dtype)
                logger.info(f".to({self.weight_dtype}) took {perf_counter()-t0:.1f}s")
            else:
                logger.info(f"Already {self.weight_dtype}, skipping .to()")

            if self.lora_init_path:
                # Standard LoRA training: use load_lora to preserve saved config
                logger.info(f"\n--------Loading LoRA weights from {self.lora_init_path}--------\n")
                t0 = perf_counter()
                transformer = load_lora(transformer, self.lora_init_path, is_trainable=True)
                logger.info(f"load_lora took {perf_counter()-t0:.1f}s")
            else:
                # No pretrained LoRA, create fresh LoRA layers
                t0 = perf_counter()
                transformer = add_lora(transformer, self.lora_rank, init_lora_weights=self.lora_init_weights)
                logger.info(f"add_lora took {perf_counter()-t0:.1f}s")
                logger.info(f"Added fresh LoRA layers (init={self.lora_init_weights})")

            # Cast trainable params
            cast_training_params(
                [transformer],
                dtype=self.weight_dtype if self.use_fsdp else torch.float32,
            )

            if hasattr(transformer, "print_trainable_parameters"):
                transformer.print_trainable_parameters()
        else:
            # No LoRA training
            if self.lora_init_path:
                # Load and merge LoRA for inference
                logger.info(f"\n--------Loading pretrained LoRA from {self.lora_init_path} and fusing--------\n")
                transformer = load_lora(transformer, self.lora_init_path, is_trainable=True)
                transformer = transformer.merge_and_unload()
            transformer.requires_grad_(True)
            model_dtype = next(transformer.parameters()).dtype
            assert model_dtype == torch.float32, f"Expected float32, got {model_dtype}"

        return transformer

    def _log_param_counts(self, transformer: nn.Module) -> None:
        """Log parameter counts."""
        param_count = sum(p.numel() for p in transformer.parameters()) / 10**9
        trainable_param_count = sum(
            p.numel() for p in transformer.parameters() if p.requires_grad
        ) / 10**9
        logger.info(f"Transformer Parameters: {param_count}B")
        logger.info(f"Trainable transformer Parameters: {trainable_param_count}B")


def init_transformer(
    accelerator: Accelerator,
    transformer_config: Dict[str, Any],
    lora_rank: int = 0,
    use_fsdp: bool = False,
    weight_dtype: torch.dtype = torch.float32,
    gradient_checkpointing: bool = False,
    debug: bool = False,
    lora_init_weights: Union[bool, str] = True,
    use_varlen_attention: bool = False,
    parallel_checkpoint_load: bool = True,
    transformer_init_path: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    checkpoint_local_path: Optional[str] = None,
) -> TransformerInitResult:
    """
    Convenience function for initializing a transformer.

    Args:
        accelerator: HuggingFace Accelerator instance
        transformer_config: Transformer configuration dict
        lora_rank: LoRA rank (0 for no LoRA)
        use_fsdp: Whether to use FSDP
        weight_dtype: Weight dtype for model
        gradient_checkpointing: Enable gradient checkpointing
        debug: Debug mode (skip weight loading)
        lora_init_weights: LoRA initialization method
            - True (default): Standard init, A=Kaiming, B=zeros
            - "gaussian": Both A and B are Gaussian random
        use_varlen_attention: Convert to varlen attention transformer variant
        parallel_checkpoint_load: Enable parallel checkpoint loading across all FSDP ranks.
            When True (default), all ranks load weights simultaneously. Set to False
            to revert to rank-0-only loading if needed.
        resume_from_checkpoint: See TransformerInitHandler's own docstring -- optional, only
            meaningful for LoRA (lora_rank > 0); default None is a full no-op.
        checkpoint_local_path: See TransformerInitHandler's own docstring.

    Returns:
        TransformerInitResult with initialized models
    """
    handler = TransformerInitHandler(
        accelerator=accelerator,
        transformer_config=transformer_config,
        lora_rank=lora_rank,
        use_fsdp=use_fsdp,
        weight_dtype=weight_dtype,
        gradient_checkpointing=gradient_checkpointing,
        debug=debug,
        lora_init_weights=lora_init_weights,
        use_varlen_attention=use_varlen_attention,
        parallel_checkpoint_load=parallel_checkpoint_load,
        transformer_init_path=transformer_init_path,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_local_path=checkpoint_local_path,
    )
    return handler.initialize()
