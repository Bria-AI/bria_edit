"""
Varlen attention variant of BriaTransformer2DModel for FSDP-compatible training.

This module provides transformer block subclasses that override forward() to handle
variable-length sequences with varlen attention. By subclassing the original blocks:
1. FSDP wrapping works correctly (they ARE FluxTransformerBlock subclasses)
2. Calling self.norm1(...) goes through proper nn.Module forward hooks
3. State dict loads directly (same submodule names)

Key insight: Flux joint transformer blocks have separate paths for text vs image:
- norm1 + ff for image tokens
- norm1_context + ff_context for text tokens
- Combined only for attention

This approach eliminates padding overhead when text sequences have varying lengths.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from torch.nn.attention.varlen import varlen_attn
from transformer_bria_repa import Bria4Transformer2DModel, FluxSingleTransformerBlock

logger = logging.get_logger(__name__)

# Compiled varlen attention for torch.compile compatibility
# Using max-autotune-no-cudagraphs for best performance without CUDA graph overhead
_compiled_varlen_attn: Optional[Callable] = None


def get_compiled_varlen_attn() -> Callable:
    """Get or create compiled varlen_attn function (lazy initialization)."""
    global _compiled_varlen_attn
    if _compiled_varlen_attn is None:
        _compiled_varlen_attn = torch.compile(varlen_attn, mode="max-autotune-no-cudagraphs")
    return _compiled_varlen_attn


# =============================================================================
# Helper Functions for Interleave/Split Operations
# =============================================================================


def build_interleave_indices(
    text_lengths: List[int],
    img_lengths: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute indices for interleaving text and image tokens.

    Call this ONCE per forward pass, outside compiled regions.
    The indices can then be reused for all interleave/split operations.

    Args:
        text_lengths: List of text lengths per sample
        img_lengths: List of image lengths per sample
        device: Device for output tensors

    Returns:
        text_indices: [total_text] - where each text token goes in interleaved output
        img_indices: [total_img] - where each image token goes in interleaved output
    """
    text_indices = []
    img_indices = []
    output_offset = 0

    for t_len, i_len in zip(text_lengths, img_lengths):
        # Text tokens go first in each sample's segment
        text_indices.append(torch.arange(output_offset, output_offset + t_len, device=device))
        output_offset += t_len
        # Image tokens follow
        img_indices.append(torch.arange(output_offset, output_offset + i_len, device=device))
        output_offset += i_len

    return torch.cat(text_indices), torch.cat(img_indices)


def interleave_with_indices(
    text_tokens: torch.Tensor,  # [total_text, ...]
    img_tokens: torch.Tensor,  # [total_img, ...]
    text_indices: torch.Tensor,  # [total_text]
    img_indices: torch.Tensor,  # [total_img]
) -> torch.Tensor:
    """Interleave using precomputed indices. Compiler-friendly.

    Args:
        text_tokens: Flat text tensor [total_text, ...]
        img_tokens: Flat image tensor [total_img, ...]
        text_indices: Precomputed positions for text tokens
        img_indices: Precomputed positions for image tokens

    Returns:
        Interleaved tensor [total_seq, ...]
    """
    total_seq = text_tokens.shape[0] + img_tokens.shape[0]
    output_shape = (total_seq,) + text_tokens.shape[1:]
    output = torch.empty(output_shape, device=text_tokens.device, dtype=text_tokens.dtype)
    output[text_indices] = text_tokens
    output[img_indices] = img_tokens
    return output


def split_with_indices(
    packed: torch.Tensor,  # [total_seq, ...]
    text_indices: torch.Tensor,  # [total_text]
    img_indices: torch.Tensor,  # [total_img]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split interleaved tensor using precomputed indices. Compiler-friendly.

    Args:
        packed: Interleaved tensor [total_seq, ...]
        text_indices: Precomputed positions for text tokens
        img_indices: Precomputed positions for image tokens

    Returns:
        Tuple of (text_flat, img_flat)
    """
    return packed[text_indices], packed[img_indices]


# =============================================================================
# Varlen Block Subclasses
# =============================================================================


class VarlenFluxTransformerBlock(FluxTransformerBlock):
    """Joint block with varlen attention support.

    Inherits from FluxTransformerBlock to ensure FSDP auto-wrap policy recognizes it
    and state_dict keys match for checkpoint loading.
    """

    def forward(
        self,
        text_flat: torch.Tensor,  # [total_text, D]
        img_flat: torch.Tensor,  # [total_img, D]
        temb: torch.Tensor,  # [bsz, temb_dim]
        text_batch_ids: torch.Tensor,  # [total_text]
        img_batch_ids: torch.Tensor,  # [total_img]
        text_indices: torch.Tensor,  # [total_text] precomputed interleave indices
        img_indices: torch.Tensor,  # [total_img] precomputed interleave indices
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,  # [bsz + 1]
        max_seqlen: int,
        attn_fn: Optional[Callable] = None,  # Optional attention function (default: varlen_attn)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with flat text/image tensors using varlen attention.

        Calls self.* modules directly (not block.*) to ensure FSDP parameter gathering.
        Uses precomputed indices for compiler-friendly interleave/split operations.

        Args:
            attn_fn: Optional attention function. If None, uses varlen_attn.
                     Pass get_compiled_varlen_attn() for compiled version.
        """
        if attn_fn is None:
            attn_fn = varlen_attn
        dtype = text_flat.dtype

        # === Expand temb per-token ===
        text_temb = temb[text_batch_ids]  # [total_text, temb_dim]
        img_temb = temb[img_batch_ids]  # [total_img, temb_dim]

        # === AdaLayerNormZero for text ===
        # self.norm1_context is AdaLayerNormZero with .linear, .silu, .norm
        text_emb = self.norm1_context.linear(self.norm1_context.silu(text_temb))
        c_shift, c_scale, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = text_emb.chunk(6, dim=-1)
        norm_text = self.norm1_context.norm(text_flat) * (1 + c_scale) + c_shift

        # === AdaLayerNormZero for image ===
        img_emb = self.norm1.linear(self.norm1.silu(img_temb))
        shift, scale, gate_msa, shift_mlp, scale_mlp, gate_mlp = img_emb.chunk(6, dim=-1)
        norm_img = self.norm1.norm(img_flat) * (1 + scale) + shift

        # === Attention projections ===
        attn = self.attn
        inner_dim = attn.to_q.out_features
        head_dim = inner_dim // attn.heads

        # Text uses add_q/k/v_proj
        text_q = attn.add_q_proj(norm_text)
        text_k = attn.add_k_proj(norm_text)
        text_v = attn.add_v_proj(norm_text)

        # Image uses to_q/k/v
        img_q = attn.to_q(norm_img)
        img_k = attn.to_k(norm_img)
        img_v = attn.to_v(norm_img)

        # Reshape for multi-head attention: [n_tokens, inner_dim] -> [n_tokens, heads, head_dim]
        text_q = text_q.view(-1, attn.heads, head_dim)
        text_k = text_k.view(-1, attn.heads, head_dim)
        text_v = text_v.view(-1, attn.heads, head_dim)
        img_q = img_q.view(-1, attn.heads, head_dim)
        img_k = img_k.view(-1, attn.heads, head_dim)
        img_v = img_v.view(-1, attn.heads, head_dim)

        # Apply QK norms if present
        if attn.norm_added_q is not None:
            text_q = attn.norm_added_q(text_q)
        if attn.norm_added_k is not None:
            text_k = attn.norm_added_k(text_k)
        if attn.norm_q is not None:
            img_q = attn.norm_q(img_q)
        if attn.norm_k is not None:
            img_k = attn.norm_k(img_k)

        # === Interleave Q/K/V for attention (using precomputed indices) ===
        packed_q = interleave_with_indices(text_q, img_q, text_indices, img_indices)
        packed_k = interleave_with_indices(text_k, img_k, text_indices, img_indices)
        packed_v = interleave_with_indices(text_v, img_v, text_indices, img_indices)

        # === Apply rotary embeddings ===
        if image_rotary_emb is not None:
            # apply_rotary_emb expects [bsz, heads, seq, dim]
            packed_q = packed_q.unsqueeze(0).transpose(1, 2)  # [1, heads, total_seq, head_dim]
            packed_k = packed_k.unsqueeze(0).transpose(1, 2)
            packed_q = apply_rotary_emb(packed_q, image_rotary_emb)
            packed_k = apply_rotary_emb(packed_k, image_rotary_emb)
            packed_q = packed_q.transpose(1, 2).squeeze(0)  # [total_seq, heads, head_dim]
            packed_k = packed_k.transpose(1, 2).squeeze(0)

        # === Varlen attention ===
        # Flash attention requires fp16 or bf16
        compute_dtype = torch.bfloat16 if packed_q.dtype == torch.float32 else packed_q.dtype
        attn_out = attn_fn(
            packed_q.to(compute_dtype),
            packed_k.to(compute_dtype),
            packed_v.to(compute_dtype),
            cu_seq_q=cu_seqlens,
            cu_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
            is_causal=False,
        )
        attn_out = attn_out.to(dtype)
        # attn_out: [total_seq, heads, head_dim]

        # === Split back to text and image (using precomputed indices) ===
        # Attention has separate output projections: to_add_out for text, to_out for image
        attn_out = attn_out.reshape(-1, inner_dim)
        text_attn_out, img_attn_out = split_with_indices(attn_out, text_indices, img_indices)

        # Apply text output projection (single Linear)
        text_attn_out = attn.to_add_out(text_attn_out)

        # Apply image output projection (Linear + Dropout)
        img_attn_out = attn.to_out[0](img_attn_out)
        img_attn_out = attn.to_out[1](img_attn_out)

        # === Residual + gate for attention ===
        text_flat = text_flat + c_gate_msa * text_attn_out
        img_flat = img_flat + gate_msa * img_attn_out

        # === Norm2 + FF for text ===
        text_norm2 = self.norm2_context(text_flat)
        text_ff_in = text_norm2 * (1 + c_scale_mlp) + c_shift_mlp
        text_ff_out = self.ff_context(text_ff_in)
        text_flat = text_flat + c_gate_mlp * text_ff_out

        # === Norm2 + FF for image ===
        img_norm2 = self.norm2(img_flat)
        img_ff_in = img_norm2 * (1 + scale_mlp) + shift_mlp
        img_ff_out = self.ff(img_ff_in)
        img_flat = img_flat + gate_mlp * img_ff_out

        return text_flat, img_flat


class VarlenFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    """Single block with varlen attention support.

    Inherits from FluxSingleTransformerBlock to ensure FSDP auto-wrap policy recognizes it
    and state_dict keys match for checkpoint loading.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,  # [total_seq, D] interleaved
        temb: torch.Tensor,  # [bsz, temb_dim]
        batch_ids: torch.Tensor,  # [total_seq]
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,  # [bsz + 1]
        max_seqlen: int,
        attn_fn: Optional[Callable] = None,  # Optional attention function (default: varlen_attn)
    ) -> torch.Tensor:
        """Forward pass with varlen attention.

        Calls self.* modules directly (not block.*) to ensure FSDP parameter gathering.

        Args:
            attn_fn: Optional attention function. If None, uses varlen_attn.
                     Pass get_compiled_varlen_attn() for compiled version.
        """
        if attn_fn is None:
            attn_fn = varlen_attn
        dtype = hidden_states.dtype
        residual = hidden_states

        # === Expand temb per-token ===
        temb_expanded = temb[batch_ids]  # [total_seq, temb_dim]

        # === AdaLayerNormZeroSingle: compute scale, shift, gate per-token ===
        emb = self.norm.linear(self.norm.silu(temb_expanded))  # [total_seq, 3*D]
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=-1)  # each [total_seq, D]

        # Apply norm with per-token scale/shift
        norm_hidden_states = self.norm.norm(hidden_states) * (1 + scale_msa) + shift_msa

        # === MLP branch ===
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        # === Attention branch ===
        attn = self.attn
        inner_dim = attn.to_q.out_features
        head_dim = inner_dim // attn.heads

        # Q/K/V projections
        query = attn.to_q(norm_hidden_states)  # [total_seq, inner_dim]
        key = attn.to_k(norm_hidden_states)
        value = attn.to_v(norm_hidden_states)

        # Reshape for attention: [total_seq, heads, head_dim]
        query = query.view(-1, attn.heads, head_dim)
        key = key.view(-1, attn.heads, head_dim)
        value = value.view(-1, attn.heads, head_dim)

        # Apply QK norms if present
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # === Apply rotary embeddings ===
        if image_rotary_emb is not None:
            # apply_rotary_emb expects [bsz, heads, seq, dim]
            query = query.unsqueeze(0).transpose(1, 2)  # [1, heads, total_seq, head_dim]
            key = key.unsqueeze(0).transpose(1, 2)
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
            query = query.transpose(1, 2).squeeze(0)  # [total_seq, heads, head_dim]
            key = key.transpose(1, 2).squeeze(0)

        # === Varlen attention ===
        compute_dtype = torch.bfloat16 if query.dtype == torch.float32 else query.dtype
        attn_output = attn_fn(
            query.to(compute_dtype),
            key.to(compute_dtype),
            value.to(compute_dtype),
            cu_seq_q=cu_seqlens,
            cu_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
            is_causal=False,
        )
        attn_output = attn_output.to(dtype)

        # Reshape: [total_seq, heads, head_dim] -> [total_seq, inner_dim]
        attn_output = attn_output.reshape(-1, inner_dim)

        # === Combine attention and MLP outputs ===
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)  # [total_seq, inner_dim + mlp_dim]

        # === Output projection with per-token gating ===
        hidden_states = gate_msa * self.proj_out(hidden_states)

        # === Residual connection ===
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


# =============================================================================
# Varlen Model
# =============================================================================


class BriaTransformer2DModelVarlen(nn.Module):
    """Varlen attention transformer using VarlenFlux*Block subclasses.

    Uses variable-length attention (flash_attn varlen API) to support:
    - Variable text lengths without padding overhead
    - Multi-resolution batches with different H*W per sample
    - Any batch size (not limited to bsz=1 like flash attention)

    Not a wrapper - creates its own block instances that inherit from the
    original block classes. This ensures:
    1. FSDP auto-wrap policy works (blocks ARE FluxTransformerBlock subclasses)
    2. State dict keys match for checkpoint loading
    3. No wrapper complexity
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = None,
        guidance_embeds: bool = False,
        axes_dims_rope: List[int] = [16, 56, 56],
        rope_theta: int = 10000,
        time_theta: int = 10000,
        dino_embedding_dim: int = 1024,
        repa_projector_dim: int = 2048,
        flag_repa: bool = False,
        text_encoder_dim: int = 2048,
        num_checkpointing_blocks: int = 100,
    ):
        super().__init__()

        # Store config for from_pretrained_standard
        self.config = {
            "patch_size": patch_size,
            "in_channels": in_channels,
            "num_layers": num_layers,
            "num_single_layers": num_single_layers,
            "attention_head_dim": attention_head_dim,
            "num_attention_heads": num_attention_heads,
            "joint_attention_dim": joint_attention_dim,
            "pooled_projection_dim": pooled_projection_dim,
            "guidance_embeds": guidance_embeds,
            "axes_dims_rope": axes_dims_rope,
            "rope_theta": rope_theta,
            "time_theta": time_theta,
            "dino_embedding_dim": dino_embedding_dim,
            "repa_projector_dim": repa_projector_dim,
            "flag_repa": flag_repa,
            "text_encoder_dim": text_encoder_dim,
            "num_checkpointing_blocks": num_checkpointing_blocks,
        }

        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # Import components from Bria4 model
        from bria_utils import FluxPosEmbed as EmbedND
        from diffusers.models.normalization import AdaLayerNormContinuous
        from transformer_bria_repa import TextProjection, TimestepProjEmbeddings

        self.pos_embed = EmbedND(theta=rope_theta, axes_dim=axes_dims_rope)
        self.time_embed = TimestepProjEmbeddings(embedding_dim=self.inner_dim, time_theta=time_theta)

        if guidance_embeds:
            self.guidance_embed = TimestepProjEmbeddings(embedding_dim=self.inner_dim, time_theta=time_theta)

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        # Use VarlenFluxTransformerBlock instead of FluxTransformerBlock
        self.transformer_blocks = nn.ModuleList(
            [
                VarlenFluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Use VarlenFluxSingleTransformerBlock instead of FluxSingleTransformerBlock
        self.single_transformer_blocks = nn.ModuleList(
            [
                VarlenFluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.num_checkpointing_blocks = num_checkpointing_blocks
        self.attn_fn = varlen_attn  # Default; use set_compiled_varlen() to switch

        self.flag_repa = flag_repa
        self.dino_embedding_dim = dino_embedding_dim
        self.repa_projector_dim = repa_projector_dim
        self.repa_mlp = nn.Sequential(
            nn.Linear(self.inner_dim, repa_projector_dim),
            nn.SiLU(),
            nn.Linear(repa_projector_dim, repa_projector_dim),
            nn.SiLU(),
            nn.Linear(repa_projector_dim, dino_embedding_dim),
        )

        caption_projection = [
            TextProjection(in_features=text_encoder_dim, hidden_size=self.inner_dim // 2)
            for _ in range(num_layers + num_single_layers)
        ]
        self.caption_projection = nn.ModuleList(caption_projection)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def set_compiled_varlen(self, use_compiled: bool):
        """Set whether to use compiled varlen attention for speedup."""
        self.attn_fn = get_compiled_varlen_attn() if use_compiled else varlen_attn

    @classmethod
    def from_pretrained_standard(cls, model: Bria4Transformer2DModel) -> "BriaTransformer2DModelVarlen":
        """Load from a standard Bria4Transformer2DModel checkpoint.

        Args:
            model: A Bria4Transformer2DModel with loaded weights

        Returns:
            BriaTransformer2DModelVarlen with the same weights
        """
        import inspect

        # Get valid kwargs for the constructor
        init_signature = inspect.signature(cls.__init__)
        valid_params = set(init_signature.parameters.keys()) - {"self"}

        # Filter config to only include valid parameters
        filtered_config = {k: v for k, v in model.config.items() if k in valid_params}

        # Create new instance with filtered config
        varlen_model = cls(**filtered_config)

        # Load state dict - keys match because VarlenFlux*Block inherits from Flux*Block
        varlen_model.load_state_dict(model.state_dict(), strict=False)

        return varlen_model

    # Alias for backward compatibility
    from_standard_model = from_pretrained_standard

    def forward(
        self,
        text_flat: torch.Tensor,  # [total_text, text_dim] - NOT projected
        img_flat: torch.Tensor,  # [total_img, img_dim] - NOT projected
        timestep: torch.LongTensor,
        text_lengths: List[int],
        img_lengths: List[int],
        txt_ids: torch.Tensor,  # [total_text, 3]
        img_ids: torch.Tensor,  # [total_img, 3]
        text_batch_ids: torch.Tensor,  # [total_text]
        img_batch_ids: torch.Tensor,  # [total_img]
        text_encoder_layers: Optional[List[torch.Tensor]] = None,  # List of [total_text, layer_dim]
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        actual_text_lengths: Optional[List[int]] = None,  # Actual (pre-padding) lengths for cu_seqlens
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """Forward pass with varlen attention.

        Args:
            text_flat: [total_text, text_dim] flat text embeddings (NOT projected)
            img_flat: [total_img, img_dim] flat image+context latents (NOT projected)
            timestep: [bsz] timesteps
            text_lengths: List of text sequence lengths per sample (for interleave indices)
            img_lengths: List of image+context lengths per sample
            txt_ids: [total_text, 3] position IDs for text tokens
            img_ids: [total_img, 3] position IDs for all image tokens
            text_batch_ids: [total_text] batch ID for each text token
            img_batch_ids: [total_img] batch ID for each image token
            text_encoder_layers: Optional list of [total_text, layer_dim] per-block text layers
            guidance: Optional guidance values
            joint_attention_kwargs: Additional attention kwargs
            return_dict: Whether to return dict or tuple
            actual_text_lengths: When using constant padding, these are the ACTUAL (pre-padding)
                                text lengths for cu_seqlens computation. If None, uses text_lengths.

        Returns:
            Transformer2DModelOutput with image predictions
        """
        bsz = len(text_lengths)
        device = text_flat.device
        dtype = text_flat.dtype

        # Determine text lengths for cu_seqlens (actual lengths, not padded)
        cu_seqlen_text_lengths = actual_text_lengths if actual_text_lengths is not None else text_lengths

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        # === Project inputs to inner_dim ===
        text_flat = self.context_embedder(text_flat)  # [total_text, D]
        img_flat = self.x_embedder(img_flat)  # [total_img, D]

        # === Project text encoder layers if provided ===
        if text_encoder_layers is not None:
            projected_layers = []
            for i, layer in enumerate(text_encoder_layers):
                projected_layers.append(self.caption_projection[i](layer))
            text_encoder_layers = projected_layers

        # === Compute time embedding ===
        timestep = timestep.to(dtype)
        temb = self.time_embed(timestep, dtype=dtype)

        if guidance is not None:
            guidance = guidance.to(dtype)
            temb = temb + self.guidance_embed(guidance, dtype=dtype)

        # === Precompute interleave indices ONCE (reused by all blocks) ===
        # Uses text_lengths (padded if constant padding enabled) for constant tensor shapes
        text_indices, img_indices = build_interleave_indices(text_lengths, img_lengths, device)

        # === Build packed position IDs and compute rotary embeddings ===
        packed_ids = interleave_with_indices(txt_ids, img_ids, text_indices, img_indices)
        image_rotary_emb = self.pos_embed(packed_ids)

        # === Compute cu_seqlens for varlen attention ===
        # Uses actual (non-padded) text lengths for correct attention boundaries
        seq_lengths = [t + i for t, i in zip(cu_seqlen_text_lengths, img_lengths)]
        cu_seqlens = torch.zeros(bsz + 1, dtype=torch.int32, device=device)
        for i, seq_len in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + seq_len
        max_seqlen = max(seq_lengths)

        # === Select attention function ===
        attn_fn = self.attn_fn

        # === Process through joint blocks ===
        block_id = 0
        for block in self.transformer_blocks:
            # Inject text encoder layer: replace second half of text_flat with current layer
            if text_encoder_layers is not None:
                current_layer = text_encoder_layers[block_id]
                text_flat = torch.cat([text_flat[:, : self.inner_dim // 2], current_layer], dim=-1)
            block_id += 1

            if self.training and self.gradient_checkpointing and block_id <= self.num_checkpointing_blocks:

                def create_custom_forward(module, fn):
                    def custom_forward(txt, img, t, t_batch, i_batch, t_idx, i_idx, rot_emb, cu_lens, max_len):
                        return module(txt, img, t, t_batch, i_batch, t_idx, i_idx, rot_emb, cu_lens, max_len, fn)

                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                text_flat, img_flat = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, attn_fn),
                    text_flat,
                    img_flat,
                    temb,
                    text_batch_ids,
                    img_batch_ids,
                    text_indices,
                    img_indices,
                    image_rotary_emb,
                    cu_seqlens,
                    max_seqlen,
                    **ckpt_kwargs,
                )
            else:
                text_flat, img_flat = block(
                    text_flat=text_flat,
                    img_flat=img_flat,
                    temb=temb,
                    text_batch_ids=text_batch_ids,
                    img_batch_ids=img_batch_ids,
                    text_indices=text_indices,
                    img_indices=img_indices,
                    image_rotary_emb=image_rotary_emb,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    attn_fn=attn_fn,
                )

        # Build combined batch_ids for the interleaved sequence (used in single blocks)
        combined_batch_ids = (
            interleave_with_indices(
                text_batch_ids.unsqueeze(-1).float(),
                img_batch_ids.unsqueeze(-1).float(),
                text_indices,
                img_indices,
            )
            .squeeze(-1)
            .long()
        )

        # === Process through single blocks ===
        for block in self.single_transformer_blocks:
            # Inject text encoder layer
            if text_encoder_layers is not None:
                current_layer = text_encoder_layers[block_id]
                text_flat = torch.cat([text_flat[:, : self.inner_dim // 2], current_layer], dim=-1)
            block_id += 1

            # Interleave for this block: [text_0, img_0, text_1, img_1, ...]
            hidden_states = interleave_with_indices(text_flat, img_flat, text_indices, img_indices)

            if self.training and self.gradient_checkpointing and block_id <= self.num_checkpointing_blocks:

                def create_custom_forward(module, fn):
                    def custom_forward(hs, t, b_ids, rot_emb, cu_lens, max_len):
                        return module(hs, t, b_ids, rot_emb, cu_lens, max_len, fn)

                    return custom_forward

                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, attn_fn),
                    hidden_states,
                    temb,
                    combined_batch_ids,
                    image_rotary_emb,
                    cu_seqlens,
                    max_seqlen,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    batch_ids=combined_batch_ids,
                    image_rotary_emb=image_rotary_emb,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    attn_fn=attn_fn,
                )

            # Split back to separate text and image for next iteration
            text_flat, img_flat = split_with_indices(hidden_states, text_indices, img_indices)

        # === Extract image portions ===
        img_out_flat = img_flat  # [total_img, D]

        # === Final norm and projection (flat) ===
        # AdaLayerNormContinuous expects [bsz, seq, dim] and [bsz, temb_dim], but we have flat tokens.
        # Expand temb per-token (same pattern as in VarlenFluxTransformerBlock/VarlenFluxSingleTransformerBlock)
        # and compute scale/shift manually to avoid the [:, None, :] broadcasting in norm_out.forward()
        temb_expanded = temb[img_batch_ids]  # [total_img, temb_dim]
        emb = self.norm_out.linear(self.norm_out.silu(temb_expanded))  # [total_img, 2*D]
        scale, shift = torch.chunk(emb, 2, dim=1)  # each [total_img, D]
        output_flat = self.norm_out.norm(img_out_flat) * (1 + scale) + shift  # [total_img, D]
        output = self.proj_out(output_flat)  # [total_img, out_channels]

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @property
    def attn_processors(self) -> Dict[str, Any]:
        """Get dict of attention processors keyed by module path."""
        processors = {}

        def fn_recursive_get(name: str, module: nn.Module):
            if hasattr(module, "processor"):
                processors[f"{name}.processor"] = module.processor
            for sub_name, child in module.named_children():
                fn_recursive_get(f"{name}.{sub_name}", child)

        for name, module in self.named_children():
            fn_recursive_get(name, module)

        return processors

    def set_attn_processor(self, processor) -> None:
        """Set attention processor for all attention layers."""

        def fn_recursive_set(module: nn.Module, proc):
            if hasattr(module, "set_processor"):
                module.set_processor(proc)
            for child in module.children():
                fn_recursive_set(child, proc)

        for module in self.children():
            fn_recursive_set(module, processor)
