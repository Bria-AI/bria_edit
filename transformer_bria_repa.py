from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin, FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.embeddings import TimestepEmbedding, get_timestep_embedding
from bria_utils import FluxPosEmbed as EmbedND
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import (
    Attention,
    FluxAttnProcessor2_0,
    FluxAttnProcessor2_0_NPU,
)
from diffusers.utils import deprecate
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.models.normalization import AdaLayerNormZeroSingle

logger = logging.get_logger(__name__)

@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        if is_torch_npu_available():
            deprecation_message = (
                "Defaulting to FluxAttnProcessor2_0_NPU for NPU devices will be removed. Attention processors "
                "should be set explicitly using the `set_attn_processor` method."
            )
            deprecate("npu_processor", "0.34.0", deprecation_message)
            processor = FluxAttnProcessor2_0_NPU()
        else:
            processor = FluxAttnProcessor2_0()

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states
    
class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)

    def forward(self, caption):
        hidden_states = self.linear(caption)
        return hidden_states


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1,time_theta=10000):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.time_theta=time_theta

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
            max_period=self.time_theta
        )
        return t_emb
    
class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, time_theta):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0,time_theta=time_theta)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        
    def forward(self, timestep, dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=dtype))  # (N, D)
        return timesteps_emb
    
class Bria4Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
        ...
    """

    _supports_gradient_checkpointing = True

    @register_to_config
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
        rope_theta = 10000,
        time_theta = 10000,
        dino_embedding_dim : int = 1024,
        repa_projector_dim: int = 2048,
        flag_repa = False,
        text_encoder_dim: int = 2048,
        num_checkpointing_blocks: int = 100,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = EmbedND(theta=rope_theta, axes_dim=axes_dims_rope)
        
        self.time_embed = TimestepProjEmbeddings(
            embedding_dim=self.inner_dim,time_theta=time_theta
        )

        # if pooled_projection_dim:
        #     self.pooled_text_embed = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim=self.inner_dim, act_fn="silu")
    
        if guidance_embeds:
            self.guidance_embed = TimestepProjEmbeddings(embedding_dim=self.inner_dim)
    
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )
        
        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.num_checkpointing_blocks = num_checkpointing_blocks

        self.flag_repa = flag_repa
        self.dino_embedding_dim = dino_embedding_dim
        self.repa_projector_dim = repa_projector_dim
        self.repa_mlp = nn.Sequential(
        nn.Linear(self.inner_dim, self.repa_projector_dim),
        nn.SiLU(),
        nn.Linear(self.repa_projector_dim, self.repa_projector_dim),
        nn.SiLU(),
        nn.Linear(self.repa_projector_dim, self.dino_embedding_dim),
        )

        caption_projection = [TextProjection(in_features = text_encoder_dim, hidden_size = self.inner_dim // 2) for i in range(self.config.num_layers + self.config.num_single_layers)]
        self.caption_projection = nn.ModuleList(caption_projection)


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        text_encoder_layers: list = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        controlnet_block_samples = None,
        controlnet_single_block_samples=None,

    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype)
        else:
            guidance = None

        # temb = (
        #     self.time_text_embed(timestep, pooled_projections)
        #     if guidance is None
        #     else self.time_text_embed(timestep, guidance, pooled_projections)
        # )

        temb = self.time_embed(timestep,dtype=hidden_states.dtype)

        # if pooled_projections:
        #     temb+=self.pooled_text_embed(pooled_projections)
        
        if guidance:
            temb+=self.guidance_embed(guidance,dtype=hidden_states.dtype)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if len(txt_ids.shape)==2:
            ids = torch.cat((txt_ids, img_ids), dim=0)
        else:
            ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pos_embed(ids)

        new_text_encoder_layers = []
        for i, text_encoder_layer in enumerate(text_encoder_layers):
            text_encoder_layer = self.caption_projection[i](text_encoder_layer)
            new_text_encoder_layers.append(text_encoder_layer)
        text_encoder_layers = new_text_encoder_layers
        
        block_id = 0
        for index_block, block in enumerate(self.transformer_blocks):
            current_text_encoder_layer = text_encoder_layers[block_id]
            encoder_hidden_states = torch.cat([
                encoder_hidden_states[:,:, :self.inner_dim//2], current_text_encoder_layer
            ], dim = -1)
            block_id += 1
            if torch.is_grad_enabled() and self.gradient_checkpointing and block_id < self.num_checkpointing_blocks:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs
                )
            
            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        repa_enc = self.repa_mlp(hidden_states) if self.flag_repa else None

        for index_block, block in enumerate(self.single_transformer_blocks):
            current_text_encoder_layer = text_encoder_layers[block_id]
            encoder_hidden_states = torch.cat([
                encoder_hidden_states[:,:, :self.inner_dim//2], current_text_encoder_layer
            ], dim = -1)
            block_id += 1
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            if torch.is_grad_enabled() and self.gradient_checkpointing and block_id < self.num_checkpointing_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs
                )
            
            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

            encoder_hidden_states = hidden_states[:, : encoder_hidden_states.shape[1], ...]
            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, repa_enc)

        return Transformer2DModelOutput(sample=output)
