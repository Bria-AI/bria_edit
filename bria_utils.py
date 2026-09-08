"""Training utilities for fibo_edit_next.

Merged from the source repo's bria_utils.py + bria4_utils.py -- in this
trimmed self-contained copy the two had shrunk to where the split no longer
earned its keep: bria4_utils.py's only cross-file import was get_env_prefix,
which bria_utils.py ALSO defined (identically, byte-for-byte apart from
quote style) -- a genuine duplicate, not just a similar-sounding name. One
file, one get_env_prefix. See README.md in this directory.
"""

import math
import os
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import torch
from diffusers.utils import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_env_prefix():
    env = os.environ.get("CLOUD_PROVIDER", "AWS").upper()
    if env == "AWS":
        return "SM_CHANNEL"
    elif env == "AZURE":
        return "AZUREML_DATAREFERENCE"

    raise Exception(f"Env {env} not supported")


# =============================================================================
# RoPE positional embedding (used by transformer_bria_repa.py / transformer_bria_varlen.py)
# =============================================================================


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


class FluxPosEmbed(torch.nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        freqs_dtype = torch.float32 if is_mps else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


# =============================================================================
# LR scheduler
# =============================================================================

from diffusers.optimization import get_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

# Not really cosine but with decay
def get_cosine_schedule_with_warmup_and_decay(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, constant_steps=-1,eps=1e-5
) -> LambdaLR:

    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_periods (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        constant_steps (`int`):
            The total number of constant lr steps following a warmup

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if constant_steps <=0:
        constant_steps = num_training_steps-num_warmup_steps

    def lr_lambda(current_step):
        # Accelerate sends current_step*num_processes
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step<num_warmup_steps+constant_steps:
            return 1

        # Cosine decay from 1.0 -> 0 over [warmup+constant, num_training_steps] (was linear).
        decay_span = float(max(1, num_training_steps - num_warmup_steps - constant_steps))
        progress = float(current_step - num_warmup_steps - constant_steps) / decay_span
        progress = min(1.0, max(0.0, progress))
        return max(eps, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_lr_scheduler(
        name,
        optimizer,
        num_warmup_steps,
        num_training_steps,
        constant_steps):
    if name!='constant_with_warmup_cosine_decay':
        return get_scheduler(
            name=name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)

    # Usign custom warmup+cnstant+decay scheduler
    return get_cosine_schedule_with_warmup_and_decay(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, constant_steps=constant_steps)


# =============================================================================
# Text encoder (smolLM)
# =============================================================================


@torch.no_grad()
def get_smollm_prompt_embeds(
    tokenizer: AutoTokenizer,
    text_encoder: AutoModelForCausalLM,
    prompts: Union[str, List[str]] = None,
    max_sequence_length: int = 3000,
    pad_to_length: int = None,
):
    """Encode prompts with optional constant padding for torch.compile compatibility.

    Args:
        tokenizer: The tokenizer instance
        text_encoder: The text encoder model
        prompts: Single prompt string or list of prompts
        max_sequence_length: Maximum sequence length for truncation
        pad_to_length: If set, pad all sequences to this constant length (enables torch.compile)
                       If None, uses "longest" padding (dynamic, causes recompilation)

    Returns:
        Tuple of (prompt_embeds, hidden_states, attention_mask)
    """
    prompts = [prompts] if isinstance(prompts, str) else prompts
    bot_token_id = 128000 # same as Llama

    if "" in prompts:
        bs = len(prompts)
        assert all(p == "" for p in prompts)
        text_input_ids = torch.zeros([bs, 1], dtype=torch.int64, device=text_encoder.device) + bot_token_id
        attention_mask = torch.ones([bs, 1], dtype=torch.int64, device=text_encoder.device)
    else:
        # Use constant padding if pad_to_length is specified, otherwise pad to longest
        if pad_to_length is not None:
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=pad_to_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        else:
            text_inputs = tokenizer(
                prompts,
                padding="longest",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids.to(text_encoder.device)
        attention_mask = text_inputs.attention_mask.to(text_encoder.device)

    if len(prompts) == 1 and pad_to_length is None:
        assert (attention_mask == 1).all()

    hidden_states = text_encoder(text_input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
    # We need a 4096 dim so since we have 2048 we take last 2 layers
    prompt_embeds = torch.concat([hidden_states[-1], hidden_states[-2]], dim=-1)

    return prompt_embeds, hidden_states, attention_mask


@torch.no_grad()
def get_smollm_prompt_embeds_varlen(
    tokenizer: AutoTokenizer,
    text_encoder: AutoModelForCausalLM,
    prompts: Union[str, List[str]] = None,
    max_sequence_length: int = 3000,
    pad_to_length: int = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    """Encode prompts without inter-sequence padding, return varlen embeddings + lengths.

    Instead of padding all sequences to the longest, this encodes each prompt
    separately and concatenates the results. This eliminates padding overhead
    in attention computation when used with varlen attention.

    When pad_to_length is specified, each individual sequence is padded to that
    constant length before concatenation. This keeps total_tokens constant across
    batches (= bsz * pad_to_length), enabling torch.compile without recompilation.
    The returned text_lengths still contain the ACTUAL (pre-padding) lengths for
    cu_seqlens computation.

    Args:
        tokenizer: The tokenizer instance
        text_encoder: The text encoder model
        prompts: Single prompt string or list of prompts
        max_sequence_length: Maximum sequence length for truncation
        pad_to_length: If set, pad each sequence to this constant length (enables torch.compile)
                       If None, no padding (dynamic total_tokens, causes recompilation)

    Returns:
        Tuple of:
        - varlen_embeddings: [total_tokens, dim] concatenated embeddings
        - varlen_layers: List of [total_tokens, layer_dim] for each hidden layer
        - text_lengths: List of ACTUAL sequence lengths for each prompt (for cu_seqlens)
    """
    prompts = [prompts] if isinstance(prompts, str) else prompts
    bot_token_id = 128000  # same as Llama

    all_embeddings = []
    all_layers_dict: dict = {}  # layer_idx -> list of tensors
    text_lengths = []  # Track ACTUAL lengths (pre-padding) for cu_seqlens

    for prompt in prompts:
        if prompt == "":
            # Empty prompt: use bot token
            text_input_ids = torch.zeros([1, 1], dtype=torch.int64, device=text_encoder.device) + bot_token_id
        else:
            # Tokenize single prompt (no padding needed at tokenization)
            text_inputs = tokenizer(
                prompt,
                padding=False,
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(text_encoder.device)

        # Encode single prompt. Pass an explicit all-ones mask so HF takes the same
        # attention-kernel path as the flash encoder / eval pipeline: without it, bf16
        # embeddings differ ~1.4% (deterministic kernel-route rounding) — a train/eval
        # skew unique to varlen runs.
        attention_mask = torch.ones_like(text_input_ids)
        hidden_states = text_encoder(
            text_input_ids, attention_mask=attention_mask, output_hidden_states=True
        ).hidden_states

        # Collect embeddings: concat last two layers for 4096 dim
        prompt_embeds = torch.concat([hidden_states[-1], hidden_states[-2]], dim=-1)
        prompt_embeds = prompt_embeds.squeeze(0)  # [seq_len, dim]
        actual_len = prompt_embeds.shape[0]
        text_lengths.append(actual_len)  # Track ACTUAL length for cu_seqlens

        # Pad to constant length if specified
        if pad_to_length is not None and actual_len < pad_to_length:
            padding = torch.zeros(
                pad_to_length - actual_len,
                prompt_embeds.shape[-1],
                dtype=prompt_embeds.dtype,
                device=prompt_embeds.device,
            )
            prompt_embeds = torch.cat([prompt_embeds, padding], dim=0)

        all_embeddings.append(prompt_embeds)  # [padded_seq_len, dim]

        # Collect all hidden layers and pad them similarly
        for i, layer in enumerate(hidden_states):
            layer_embeds = layer.squeeze(0)  # [seq_len, layer_dim]

            # Pad layer to constant length if specified
            if pad_to_length is not None and actual_len < pad_to_length:
                layer_padding = torch.zeros(
                    pad_to_length - actual_len,
                    layer_embeds.shape[-1],
                    dtype=layer_embeds.dtype,
                    device=layer_embeds.device,
                )
                layer_embeds = torch.cat([layer_embeds, layer_padding], dim=0)

            if i not in all_layers_dict:
                all_layers_dict[i] = []
            all_layers_dict[i].append(layer_embeds)  # [padded_seq_len, layer_dim]

    # Concatenate all embeddings (varlen, constant size per sample if pad_to_length)
    varlen_embeddings = torch.cat(all_embeddings, dim=0)  # [total_tokens, dim]

    # Concatenate all layers
    num_layers = len(all_layers_dict)
    varlen_layers = [torch.cat(all_layers_dict[i], dim=0) for i in range(num_layers)]

    return varlen_embeddings, varlen_layers, text_lengths


def pad_embedding(prompt_embeds, max_tokens):
    # Padds a tensor which is not masked, i.e. the "initial" tensor mask is 1's
    # We extend the tokens to max tokens and provide a mask to differentiate the masked areas
    b, seq_len, dim = prompt_embeds.shape
    padding = torch.zeros((b, max_tokens - seq_len, dim), dtype=prompt_embeds.dtype, device=prompt_embeds.device)
    attentions_mask = torch.zeros((b, max_tokens), dtype=prompt_embeds.dtype, device=prompt_embeds.device)
    attentions_mask[:, :seq_len] = 1  # original tensor is not masked
    prompt_embeds = torch.concat([prompt_embeds, padding], dim=1)

    return prompt_embeds, attentions_mask


def init_text_encoder(text_encoder_type, device, weight_dtype, text_encoder_path=None):
    """Load the training text encoder.

    Trimmed to smolLM only: train_fibo_edit_standard.py's validate_config() hard-rejects
    any other text_encoder_type, so the original T5/Llama branches here (and their
    get_t5_prompt_embeds/get_llama_prompt_embeds helpers) were unreachable dead code for
    this training script -- removed. See README.md in this directory.
    """
    if text_encoder_type != "smolLM":
        raise Exception(f"{text_encoder_type} not supported")

    print("Loading smolLM text encoder")
    model_name = "HuggingFaceTB/SmolLM3-3B"
    try:
        smolLM_init = text_encoder_path or os.environ.get(f"{get_env_prefix()}_TEXT_ENCODER")
        tokenizer = AutoTokenizer.from_pretrained(smolLM_init)
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(smolLM_init, torch_dtype=weight_dtype, low_cpu_mem_usage=True)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=weight_dtype, low_cpu_mem_usage=True)
            .to(device)
            .eval()
            .requires_grad_(False)
        )

    return tokenizer, text_encoder, get_smollm_prompt_embeds


def init_wandb(accelerator, args, project_name="sd-eiga", extend_name=False):
    wandb.login(key=os.environ["WANDB_TOKEN"])

    conf = {
        **vars(args),
        **{
            "accelerator_use_distributed": accelerator.use_distributed,
            "accelerator_mixed_precision": accelerator.mixed_precision,
            "accelerator_distributed_type": str(accelerator.distributed_type),
        },
    }
    # cloud = os.environ.get("CLOUD_PROVIDER",'AWS')
    if extend_name:
        wandb_name = f"{args.wandb_name}_acc_{args.gradient_accumulation_steps}_proc_{accelerator.num_processes}_enc_{args.text_encoder_type}"
    else:
        wandb_name = args.wandb_name
    wandb_name += "_" + datetime.now().strftime("%Y-%m-%d_%H-%M")
    wandb.init(
        project=project_name,
        entity="bria",
        config=conf,
        mode="online",
        name=wandb_name,
    )
    accelerator.init_trackers("text2image-fine-tune")  # config=vars(args)
    wandb.config.update(vars(args), allow_val_change=True)


# =============================================================================
# Timestep sampling (flow-matching noise schedule)
# =============================================================================


def init_training_scheduler(uniform_prob: float = 0.1):
    return ShiftedStretchedLogitNormalTimestepSampler(uniform_prob=uniform_prob)


# Kfir's timestep sampler


class TimestepSampler:
    """Base class for timestep samplers.

    Timestep samplers are used to sample timesteps for diffusion models.
    They should implement both sample() and sample_for() methods.
    """

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: (optional) Length of the sequence being processed
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    """Samples timesteps uniformly between min_value and max_value (default 0 and 1)."""

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:  # noqa: ARG002
        return torch.rand(batch_size, device=device) * (self.max_value - self.min_value) + self.min_value

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, device=batch.device)


class ShiftedLogitNormalTimestepSampler:
    """
    Samples timesteps from a shifted logit-normal distribution,
    where the shift is determined by the sequence length.
    """

    def __init__(self, std: float = 1.0):
        self.std = std

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch from a shifted logit-normal distribution.

        Args:
            batch_size: Number of timesteps to sample
            seq_length: Length of the sequence being processed, used to determine the shift
            device: Device to place the samples on

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by seq_length
        """
        shift = self._get_shift_for_sequence_length(seq_length)
        normal_samples = torch.randn((batch_size,), device=device) * self.std + shift
        sigmas = torch.sigmoid(normal_samples)
        return sigmas

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by the sequence length
            of the input batch

        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length, device=batch.device)

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 256,
        max_tokens: int = 4096,
        min_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        # Calculate the shift value for a given sequence length using linear interpolation
        # between min_shift and max_shift based on sequence length.
        m = (max_shift - min_shift) / (max_tokens - min_tokens)  # Calculate slope
        b = min_shift - m * min_tokens  # Calculate y-intercept
        shift = m * seq_length + b  # Apply linear equation y = mx + b
        return shift


class ShiftedStretchedLogitNormalTimestepSampler:
    """
    Samples timesteps from a stretched logit-normal distribution,
    where the shift is determined by the sequence length.
    """

    def __init__(self, std: float = 1.0, uniform_prob: float = 0.1):
        self.std = std
        self.shifted_logit_normal_sampler = ShiftedLogitNormalTimestepSampler(std=std)
        self.uniform_sampler = UniformTimestepSampler()
        self.uniform_prob = uniform_prob

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        # Determine which sampler to use for each batch element
        should_use_uniform = torch.rand(batch_size, device=device) < self.uniform_prob

        # Initialize an empty tensor for the results
        timesteps = torch.empty(batch_size, device=device)

        # Sample from uniform sampler where should_use_uniform is True
        num_uniform = should_use_uniform.sum().item()
        if num_uniform > 0:
            timesteps[should_use_uniform] = self.uniform_sampler.sample(
                batch_size=num_uniform, seq_length=seq_length, device=device
            )

        # Sample from shifted logit-normal sampler where should_use_uniform is False
        should_use_shifted = ~should_use_uniform
        num_shifted = should_use_shifted.sum().item()
        if num_shifted > 0:
            timesteps[should_use_shifted] = self.shifted_logit_normal_sampler.sample(
                batch_size=num_shifted, seq_length=seq_length, device=device
            )
        return timesteps

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)

        Returns:
            Tensor of shape (batch_size,) containing timesteps

        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size=batch_size, seq_length=seq_length, device=batch.device)
