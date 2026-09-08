"""
Checkpoint loading utilities for FIBO Edit training.

This module provides a unified interface for loading checkpoints across
all training configurations (LoRA/non-LoRA, FSDP/DDP).
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from accelerate import Accelerator
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


@dataclass
class CheckpointLoadResult:
    """Result of loading a checkpoint."""

    global_step: int
    checkpoint_path: Optional[str]
    optimizer_loaded: bool
    scheduler_loaded: bool
    lora_weights_loaded: bool = False


@dataclass
class CheckpointLoadConfig:
    """Configuration for checkpoint loading behavior."""

    load_optimizer: bool = True
    load_scheduler: bool = True
    step_offset: int = 2  # Offset added to step from checkpoint name
    map_location: str = "cpu"  # Device for loading state dicts


class CheckpointLoader:
    """
    Unified checkpoint loader for all training configurations.

    Handles:
    - LoRA checkpoints (optimizer.pt/scheduler.pt separately)
    - Full model checkpoints (accelerator.load_state)
    - "latest" checkpoint auto-detection
    - Graceful fallback for missing files

    Example:
        loader = CheckpointLoader(
            accelerator=accelerator,
            checkpoint_dir="/path/to/checkpoints",
            is_lora=args.lora_rank > 0,
        )
        result = loader.load(
            resume_from="latest",  # or specific path
            optimizer=optimizer,
            scheduler=lr_scheduler,
            reinit_optimizer=args.reinit_optimizer,
            reinit_scheduler=args.reinit_scheduler,
        )
        global_step = result.global_step
    """

    def __init__(
        self,
        accelerator: Accelerator,
        checkpoint_dir: str,
        is_lora: bool = False,
        config: Optional[CheckpointLoadConfig] = None,
    ):
        """
        Initialize the checkpoint loader.

        Args:
            accelerator: HuggingFace Accelerator instance
            checkpoint_dir: Base directory containing checkpoints
            is_lora: Whether this is a LoRA training run
            config: Optional configuration for load behavior
        """
        self.accelerator = accelerator
        self.checkpoint_dir = Path(checkpoint_dir)
        self.is_lora = is_lora
        self.config = config or CheckpointLoadConfig()

    def load(
        self,
        resume_from: str = "latest",
        transformer: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        reinit_optimizer: bool = False,
        reinit_scheduler: bool = False,
        lora_weights_already_loaded: bool = False,
    ) -> CheckpointLoadResult:
        """
        Load a checkpoint.

        Args:
            resume_from: "latest" for most recent, or specific checkpoint path/name
            transformer: Transformer model to load LoRA weights into (for LoRA)
            optimizer: Optimizer to restore state into (for LoRA)
            scheduler: Scheduler to restore state into (for LoRA)
            reinit_optimizer: If True, skip loading optimizer state
            reinit_scheduler: If True, skip loading scheduler state
            lora_weights_already_loaded: Set True when the caller already loaded the LoRA
                adapter weights itself via load_lora_weights_for_resume() -- BEFORE FSDP wrapped
                the transformer (see that method's docstring for why: loading LoRA weights
                AFTER FSDP-wrap can silently produce a 0-sized local shard for very small
                target layers, e.g. a final proj_out layer, on some rank -- confirmed this
                session). When True, this call skips re-loading LoRA weights (transformer may
                even be None) but still handles optimizer/scheduler/global_step as usual.

        Returns:
            CheckpointLoadResult with global_step and loading status
        """
        checkpoint_path = self._resolve_checkpoint_path(resume_from)

        if checkpoint_path is None:
            self.accelerator.print(f"Checkpoint '{resume_from}' does not exist. Starting a new training run.")
            return CheckpointLoadResult(
                global_step=0,
                checkpoint_path=None,
                optimizer_loaded=False,
                scheduler_loaded=False,
            )

        self.accelerator.print(f"Resuming from checkpoint {checkpoint_path}")

        # Extract global step from checkpoint name
        global_step = self._extract_step(checkpoint_path)

        optimizer_loaded = False
        scheduler_loaded = False
        lora_weights_loaded = False

        if self.is_lora:
            # LoRA: Load model weights first, then optimizer/scheduler
            # This ensures optimizer state is applied to the correct weight values
            if lora_weights_already_loaded:
                lora_weights_loaded = True
            elif transformer is not None:
                lora_weights_loaded = self._load_lora_weights(checkpoint_path, transformer)
            else:
                logger.warning(
                    "No transformer provided for LoRA checkpoint loading. "
                    "LoRA weights will NOT be loaded from checkpoint. "
                    "Pass transformer to load_checkpoint() to fix this."
                )

            if not reinit_optimizer and optimizer is not None and self.config.load_optimizer:
                optimizer_loaded = self._load_optimizer_state(checkpoint_path, optimizer)

            if not reinit_scheduler and scheduler is not None and self.config.load_scheduler:
                scheduler_loaded = self._load_scheduler_state(checkpoint_path, scheduler)
        else:
            # Full model: Use accelerator.load_state
            self.accelerator.load_state(
                checkpoint_path,
                map_location=self.config.map_location,
            )
            optimizer_loaded = True
            scheduler_loaded = True

        return CheckpointLoadResult(
            global_step=global_step,
            checkpoint_path=checkpoint_path,
            optimizer_loaded=optimizer_loaded,
            scheduler_loaded=scheduler_loaded,
            lora_weights_loaded=lora_weights_loaded,
        )

    def load_lora_weights_for_resume(
        self, resume_from: str, transformer: torch.nn.Module
    ) -> "tuple[bool, Optional[str]]":
        """
        Resolve the checkpoint path and load ONLY the LoRA adapter weights into `transformer`.

        Call this BEFORE FSDP wraps the transformer (e.g. from init_handler.py, right after
        add_lora() and before accelerator.prepare(transformer)) -- loading LoRA weights via
        transformer.load_adapter() AFTER FSDP has already sharded the model can silently
        produce a 0-sized *local shard* for very small target layers on some rank, since
        load_state_dict's default (non-`assign`) copy into an already-sharded parameter can
        resolve to an empty local view rather than a real size mismatch or a real load.
        Confirmed this session: the reframe LoRA's final proj_out.lora_B (48x128 = 6,144
        elements, the smallest of all 341 targeted layers -- every other layer is 100K+
        elements) hit exactly this on a resume, while every other layer loaded fine, and
        while proj_out itself had loaded and trained correctly on the ORIGINAL fresh run
        (where add_lora() naturally runs before any FSDP wrap, so no such resume-specific
        step was ever exercised before this session). Loading before FSDP-wrap mirrors that
        working fresh-run ordering and avoids the issue -- verified locally (non-FSDP) that
        transformer.load_adapter() loads this exact checkpoint's proj_out weights with
        correct shapes when the model isn't sharded yet.

        Returns:
            (loaded, checkpoint_path) -- loaded is False and checkpoint_path is None when no
            checkpoint was found to resume from (a fresh run); in that case the caller's
            later load_checkpoint() call should proceed exactly as it always has.
        """
        checkpoint_path = self._resolve_checkpoint_path(resume_from)
        if checkpoint_path is None:
            return False, None
        loaded = self._load_lora_weights(checkpoint_path, transformer)
        return loaded, checkpoint_path

    def _resolve_checkpoint_path(self, resume_from: str) -> Optional[str]:
        """
        Resolve the checkpoint path from the resume_from argument.

        Args:
            resume_from: "latest" or a specific path/checkpoint name

        Returns:
            Full path to checkpoint directory, or None if not found
        """
        if resume_from == "latest":
            return self._find_latest_checkpoint()
        else:
            # Check if it's a full path or just a checkpoint name
            if os.path.isabs(resume_from):
                path = resume_from
            else:
                path = os.path.join(self.checkpoint_dir, os.path.basename(resume_from))

            return path if os.path.exists(path) else None

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint in the checkpoint directory."""
        if not self.checkpoint_dir.exists():
            self.accelerator.print(
                f"Checkpoint path '{self.checkpoint_dir}' does not exist. Starting a new training run."
            )
            return None

        try:
            dirs = [d for d in os.listdir(self.checkpoint_dir) if d.startswith("checkpoint")]
        except OSError as e:
            logger.warning(f"Error listing checkpoint directory: {e}")
            return None

        if not dirs:
            return None

        # Sort by step number (extracted from checkpoint_XXXXXX format)
        try:
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        except (ValueError, IndexError) as e:
            logger.warning(f"Error sorting checkpoint directories: {e}")
            # Fall back to lexicographic sorting
            dirs = sorted(dirs)

        latest = dirs[-1]
        return str(self.checkpoint_dir / latest)

    def _extract_step(self, checkpoint_path: str) -> int:
        """
        Extract global step from checkpoint path name.

        Args:
            checkpoint_path: Path to checkpoint (e.g., /path/to/checkpoint_001000)

        Returns:
            Global step value (with offset applied)
        """
        checkpoint_name = os.path.basename(checkpoint_path)
        try:
            step = int(checkpoint_name.split("_")[-1])
            return step + self.config.step_offset
        except (ValueError, IndexError):
            logger.warning(f"Could not extract step from {checkpoint_name}, defaulting to 0")
            return 0

    def _load_optimizer_state(
        self,
        checkpoint_path: str,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        """
        Load optimizer state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            optimizer: Optimizer to load state into

        Returns:
            True if loaded successfully, False otherwise
        """
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")

        if os.path.exists(optimizer_path):
            logger.info(f"Loading optimizer state from {optimizer_path}")
            try:
                state_dict = torch.load(optimizer_path, map_location=self.config.map_location)
                optimizer.load_state_dict(state_dict)
                return True
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")
                return False
        else:
            logger.info(f"No optimizer state found at {optimizer_path}, starting with fresh optimizer")
            return False

    def _load_scheduler_state(
        self,
        checkpoint_path: str,
        scheduler: Any,
    ) -> bool:
        """
        Load scheduler state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            scheduler: Scheduler to load state into

        Returns:
            True if loaded successfully, False otherwise
        """
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")

        if os.path.exists(scheduler_path):
            logger.info(f"Loading scheduler state from {scheduler_path}")
            try:
                state_dict = torch.load(scheduler_path, map_location=self.config.map_location)
                scheduler.load_state_dict(state_dict)
                return True
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")
                return False
        else:
            logger.info(f"No scheduler state found at {scheduler_path}, starting with fresh scheduler")
            return False

    def _load_lora_weights(
        self,
        checkpoint_path: str,
        transformer: torch.nn.Module,
    ) -> bool:
        """
        Load LoRA weights from checkpoint.

        Uses the same approach as init_handler._load_lora_weights():
        - For PEFT-format checkpoints (with adapter_config.json), use load_adapter()
        - For raw state dicts, use load_state_dict()

        Args:
            checkpoint_path: Path to checkpoint directory
            transformer: Transformer model to load LoRA weights into

        Returns:
            True if loaded successfully, False otherwise
        """
        # Check for adapter weights file (safetensors preferred)
        adapter_safetensors = os.path.join(checkpoint_path, "adapter_model.safetensors")
        adapter_bin = os.path.join(checkpoint_path, "adapter_model.bin")
        adapter_config = os.path.join(checkpoint_path, "adapter_config.json")

        if not os.path.exists(adapter_safetensors) and not os.path.exists(adapter_bin):
            logger.warning(f"No adapter weights found at {checkpoint_path}, skipping LoRA weight loading")
            return False

        logger.info(f"Loading LoRA weights from {checkpoint_path}")
        try:
            # Check if this is a PEFT-format checkpoint (has adapter_config.json)
            has_adapter_config = os.path.exists(adapter_config)

            if has_adapter_config:
                # PEFT-format checkpoint - use PEFT's load_adapter which handles key transformations
                transformer.load_adapter(checkpoint_path, adapter_name="default")
                logger.info(f"Successfully loaded LoRA adapter from {checkpoint_path} using load_adapter")
                return True
            else:
                # Raw state dict - load directly
                if os.path.exists(adapter_safetensors):
                    state_dict = load_file(adapter_safetensors, device=str(transformer.device))
                else:
                    state_dict = torch.load(adapter_bin, map_location=transformer.device)

                missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
                logger.info(
                    f"Loaded LoRA state dict. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}"
                )
                if unexpected_keys:
                    logger.warning(f"Unexpected keys (first 5): {unexpected_keys[:5]}")
                return True

        except Exception as e:
            logger.error(f"Failed to load LoRA weights: {e}")
            raise


def load_checkpoint(
    accelerator: Accelerator,
    checkpoint_dir: str,
    resume_from: str = "latest",
    is_lora: bool = False,
    transformer: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    reinit_optimizer: bool = False,
    reinit_scheduler: bool = False,
    lora_weights_already_loaded: bool = False,
) -> int:
    """
    Convenience function for loading a checkpoint.

    This is a simplified interface that returns just the global step,
    maintaining backward compatibility with the existing API.

    Args:
        accelerator: HuggingFace Accelerator instance
        checkpoint_dir: Base directory containing checkpoints
        resume_from: "latest" or specific checkpoint path
        is_lora: Whether this is a LoRA training run
        transformer: Transformer model to load LoRA weights into (for LoRA)
        optimizer: Optimizer to restore state into (for LoRA)
        scheduler: Scheduler to restore state into (for LoRA)
        reinit_optimizer: If True, skip loading optimizer state
        reinit_scheduler: If True, skip loading scheduler state
        lora_weights_already_loaded: See CheckpointLoader.load()'s own docstring -- pass True
            when the caller already loaded the LoRA adapter weights itself (pre-FSDP-wrap, via
            load_lora_weights_for_resume()) so this call doesn't try to reload them post-wrap.

    Returns:
        int: Global step to resume from (0 if no checkpoint found)
    """
    loader = CheckpointLoader(
        accelerator=accelerator,
        checkpoint_dir=checkpoint_dir,
        is_lora=is_lora,
    )
    result = loader.load(
        resume_from=resume_from,
        transformer=transformer,
        optimizer=optimizer,
        scheduler=scheduler,
        reinit_optimizer=reinit_optimizer,
        reinit_scheduler=reinit_scheduler,
        lora_weights_already_loaded=lora_weights_already_loaded,
    )
    return result.global_step
