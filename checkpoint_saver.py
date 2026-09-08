"""
Checkpoint saving utilities for FIBO Edit training.

This module provides a unified interface for saving checkpoints across
all training configurations (LoRA/non-LoRA, FSDP/DDP).
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator

logger = logging.getLogger(__name__)


@dataclass
class CheckpointSaveConfig:
    """Configuration for checkpoint saving behavior."""
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_data_config: bool = True
    clean_state_dict_keys: bool = True  # Remove wrapper prefixes (module., ._orig_mod)


class CheckpointSaver:
    """
    Unified checkpoint saver for all training configurations.

    Handles:
    - LoRA checkpoints (save_pretrained + optimizer.pt/scheduler.pt)
    - Full model checkpoints (accelerator.save_state)
    - FSDP state dict collection
    - DDP/torch.compile wrapper cleanup

    Example:
        saver = CheckpointSaver(
            accelerator=accelerator,
            checkpoint_dir="/path/to/checkpoints",
            is_lora=args.lora_rank > 0,
        )
        saver.save(
            step=global_step,
            transformer=transformer,
            transformer_unwrapped=transformer_unwrapped,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            data_config_path=args.json_data_input_path,
        )
    """

    def __init__(
        self,
        accelerator: Accelerator,
        checkpoint_dir: str,
        is_lora: bool = False,
        config: Optional[CheckpointSaveConfig] = None,
    ):
        """
        Initialize the checkpoint saver.

        Args:
            accelerator: HuggingFace Accelerator instance
            checkpoint_dir: Base directory for saving checkpoints
            is_lora: Whether this is a LoRA training run
            config: Optional configuration for save behavior
        """
        self.accelerator = accelerator
        self.checkpoint_dir = Path(checkpoint_dir)
        self.is_lora = is_lora
        self.config = config or CheckpointSaveConfig()

    def save(
        self,
        step: int,
        transformer: torch.nn.Module,
        transformer_unwrapped: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        data_config_path: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """
        Save a checkpoint at the given step.

        Args:
            step: Current training step
            transformer: The (potentially wrapped) transformer model
            transformer_unwrapped: Unwrapped transformer for LoRA save_pretrained
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler instance
            data_config_path: Path to data configuration JSON to copy
            checkpoint_name: Optional custom checkpoint name (default: checkpoint_{step:06d})

        Returns:
            str: Path to the saved checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{step:06d}"

        save_path = self.checkpoint_dir / checkpoint_name

        if self.is_lora:
            self._save_lora_checkpoint(
                save_path=save_path,
                transformer=transformer,
                transformer_unwrapped=transformer_unwrapped,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        else:
            self._save_full_checkpoint(save_path=save_path)

        # Save data config (main process only)
        if data_config_path and self.accelerator.is_main_process and self.config.save_data_config:
            self._save_data_config(save_path, data_config_path)

        logger.info(f"Saved checkpoint to {save_path}")
        return str(save_path)

    def _save_lora_checkpoint(
        self,
        save_path: Path,
        transformer: torch.nn.Module,
        transformer_unwrapped: Optional[torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
    ) -> None:
        """Save LoRA checkpoint with separate optimizer/scheduler files."""
        # Set FSDP to FULL_STATE_DICT mode if available
        if getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
            self.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

        self.accelerator.wait_for_everyone()

        logger.info(f"Saving LoRA checkpoint to {save_path}")

        # Get and clean state dict
        state_dict = self.accelerator.get_state_dict(transformer, unwrap=False)

        if self.config.clean_state_dict_keys:
            state_dict = self._clean_state_dict_keys(state_dict)

        # Use transformer_unwrapped for save_pretrained (PEFT compatibility)
        model_to_save = transformer_unwrapped if transformer_unwrapped is not None else transformer

        model_to_save.save_pretrained(
            str(save_path),
            state_dict=state_dict,
            is_main_process=self.accelerator.is_main_process,
        )
        del state_dict

        # Save optimizer and scheduler (main process only)
        if self.accelerator.is_main_process:
            if optimizer is not None and self.config.save_optimizer:
                optimizer_path = save_path / "optimizer.pt"
                torch.save(optimizer.state_dict(), optimizer_path)
                logger.info(f"Saved optimizer state to {optimizer_path}")

            if scheduler is not None and self.config.save_scheduler:
                scheduler_path = save_path / "scheduler.pt"
                torch.save(scheduler.state_dict(), scheduler_path)
                logger.info(f"Saved scheduler state to {scheduler_path}")

    def _save_full_checkpoint(self, save_path: Path) -> None:
        """Save full model checkpoint using accelerator.save_state."""
        logger.info(f"Saving full checkpoint to {save_path}")
        self.accelerator.save_state(str(save_path))

    def _save_data_config(self, save_path: Path, data_config_path: str) -> None:
        """Copy data configuration JSON to checkpoint directory."""
        if os.path.exists(data_config_path):
            dest_path = save_path / "data_input.json"
            shutil.copy(data_config_path, dest_path)
            logger.info(f"Copied data config to {dest_path}")

    @staticmethod
    def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove DDP/torch.compile wrapper prefixes from state dict keys."""
        cleaned = {}
        for key, value in state_dict.items():
            # Remove "module." (DDP) and "._orig_mod" (torch.compile) prefixes
            clean_key = key.replace("module.", "").replace("._orig_mod", "")
            cleaned[clean_key] = value
        return cleaned


def save_checkpoint(
    accelerator: Accelerator,
    step: int,
    checkpoint_dir: str,
    transformer: torch.nn.Module,
    is_lora: bool = False,
    transformer_unwrapped: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    data_config_path: Optional[str] = None,
) -> str:
    """
    Convenience function for saving a single checkpoint.

    This is a simplified interface for cases where you don't need
    a persistent CheckpointSaver instance.

    Args:
        accelerator: HuggingFace Accelerator instance
        step: Current training step
        checkpoint_dir: Base directory for saving checkpoints
        transformer: The (potentially wrapped) transformer model
        is_lora: Whether this is a LoRA training run
        transformer_unwrapped: Unwrapped transformer for LoRA save_pretrained
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler instance
        data_config_path: Path to data configuration JSON to copy

    Returns:
        str: Path to the saved checkpoint
    """
    saver = CheckpointSaver(
        accelerator=accelerator,
        checkpoint_dir=checkpoint_dir,
        is_lora=is_lora,
    )
    return saver.save(
        step=step,
        transformer=transformer,
        transformer_unwrapped=transformer_unwrapped,
        optimizer=optimizer,
        scheduler=scheduler,
        data_config_path=data_config_path,
    )
