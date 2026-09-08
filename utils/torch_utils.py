import functools
import glob
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

import diffusers
import numpy as np
import torch
import transformers
import webdataset as wds
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration
from bria_utils import get_env_prefix
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformer_bria_repa import FluxSingleTransformerBlock, FluxTransformerBlock
from transformer_bria_varlen import VarlenFluxTransformerBlock, VarlenFluxSingleTransformerBlock


def json_to_data(rank, json_file, attach_structured_captions=True):
    """Get training data config for a given rank from JSON config file.

    Returns:
        Tuple of (training_dir, ratio, width, height, batch_size) -- same shape as
        before mixed-resolution support was added, for backward compat with every
        pre-existing caller that unpacks this positionally. `ratio` is None and
        width/height are None for a "mixed" (multi-res) row; check `width is None`
        (equivalent to the old-callers-unaware `is_multi_res` signal) if you need it.
    """
    with open(json_file, "r") as f:
        json_data = json.load(f)
    gpu_cumsum = 0
    json_sorted_keys = sorted(json_data.keys())
    curr_gpu_config = None
    curr_gpu_name = None
    for key in json_sorted_keys:
        value = json_data[key]
        if rank < gpu_cumsum + sum(value["gpu_allocation_map"]):
            curr_gpu_config = value
            curr_gpu_name = key
            break
        gpu_cumsum += sum(value["gpu_allocation_map"])

    if curr_gpu_config is None or curr_gpu_name is None:
        raise Exception(f"Rank {rank} not found in {json_file}")

    resolutions_wh = curr_gpu_config["resolutions_wh"]
    ratios = curr_gpu_config.get("ratios")  # absent for mixed-res configs; kept optional for old configs
    gpu_allocation_map = curr_gpu_config["gpu_allocation_map"]
    n = len(gpu_allocation_map)
    batch_size = curr_gpu_config["batch_size"]
    shift = gpu_cumsum
    current_sum = 0
    for i in range(n):
        current_sum += gpu_allocation_map[i]
        if rank - shift < current_sum:
            resolution_wh = resolutions_wh[i]

            is_multi_res = (resolution_wh == "mixed")
            if is_multi_res:
                # Mixed aspect ratio mode - no resolution subdirectory
                ratio, width, height = None, None, None
                training_dir = f"{os.environ[f'{get_env_prefix()}_{curr_gpu_name}'].rstrip('/')}"
            else:
                ratio = ratios[i] if ratios else None
                resolution_wh = resolution_wh.split(",")
                width = int(resolution_wh[0])
                height = int(resolution_wh[1])
                training_dir = f"{os.environ[f'{get_env_prefix()}_{curr_gpu_name}'].rstrip('/')}/{width}x{height}"

            if attach_structured_captions:
                training_dir += "/structured-captions"
            return training_dir, ratio, width, height, batch_size


def get_accelerator(args, weight_dtype):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    fsdp_plugin = None
    if args.use_fsdp:
        os.environ["ACCELERATE_USE_FSDP"] = "true"

        # Select sharding strategy based on config. Default "full" preserves the
        # original (pre-multi-node) behavior for every caller that doesn't set this
        # new attr; opt into "hybrid" explicitly (fibo_edit_next's configs do).
        sharding_strategy_name = getattr(args, "fsdp_sharding_strategy", "full")
        if sharding_strategy_name == "hybrid":
            fsdp_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            fsdp_strategy = ShardingStrategy.FULL_SHARD

        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            sharding_strategy=fsdp_strategy,  # FULL_SHARD, #SHARD_GRAD_OP,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    FluxSingleTransformerBlock,
                    FluxTransformerBlock,
                    VarlenFluxSingleTransformerBlock,
                    VarlenFluxTransformerBlock,
                },
            ),
            mixed_precision_policy=MixedPrecision(
                param_dtype=weight_dtype,
                reduce_dtype=torch.float32,
                # reduce_dtype=weight_dtype,
                # buffer_dtype=weight_dtype,
            ),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            sync_module_states=True,
            use_orig_params=True,  # Always use orig params to avoid view issues with autograd
            param_init_fn=lambda x: x.to_empty(
                device=torch.cuda.current_device(), recurse=False
            ),  # to handle meta device params before sharding.
        )
        # 30 min timeout for large model FSDP broadcasts (default is 10 min)
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=30))]
        print("Using fsdp")
    else:
        kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False)]

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=kwargs_handlers,
    )

    # Set huggingface token key if provided
    with accelerator.main_process_first():
        if accelerator.is_local_main_process:
            if os.environ.get("HF_API_TOKEN"):
                # HfFolder.save_token was removed in newer huggingface_hub (us-west-2 py313 image);
                # HF_TOKEN env is honored across all versions.
                os.environ["HF_TOKEN"] = os.environ["HF_API_TOKEN"]

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    return accelerator


def compile_transformer(transformer, transformer_unwrapped, regional=False, **compile_kwargs):
    if regional:
        # Compile IN-PLACE (nn.Module.compile) rather than wrapping with torch.compile():
        # wrapping replaces blocks with OptimizedModule, which prefixes every state_dict key
        # with "_orig_mod." — checkpoints saved that way silently part-load in any
        # strict=False consumer (the 2026-06-11 varlen "smudging": evals ran base-model
        # blocks under trained embedders). In-place compile keeps keys identical.
        for block in transformer_unwrapped.transformer_blocks:
            block.compile(**compile_kwargs)
        for block in transformer_unwrapped.single_transformer_blocks:
            block.compile(**compile_kwargs)
    else:
        transformer.compile(**compile_kwargs)


def load_dataset_from_tars(
    training_dirs: List[str],
    rank: int,
    world_size: int,
    seed: int = 0,
    slice: bool = False,
    resampled: bool = False,
):
    tar_files = []
    for train_data_dir in training_dirs:
        files = glob.glob(f"{train_data_dir}/*.tar")
        print(f"We have {len(files)} data files on {train_data_dir}")
        tar_files += files

    total = len(tar_files)

    print(f"We have {total} data files in total")

    if slice:
        print("Using slicing")
        tar_files = tar_files[rank::world_size]  # starting from rank skip world size and take object at each step

        print(f"Process {rank} will use data files {len(tar_files)} files")

    np.random.seed(seed)
    np.random.shuffle(tar_files)

    def duplicate_name_exception_handler(e):
        if type(e) == ValueError:
            return True  # happens on duplicate vhashes
        raise e

    train_dataset = wds.WebDataset(
        tar_files, nodesplitter=wds.split_by_worker, handler=duplicate_name_exception_handler, shardshuffle=True,
        resampled=resampled,
    ).decode("torch")

    return train_dataset
