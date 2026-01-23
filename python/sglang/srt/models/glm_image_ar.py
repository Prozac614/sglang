"""Inference-only GLM-Image model compatible with HuggingFace weights."""

from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sglang.srt.configs.glm_image import (
    GlmImageConfig,
    GlmImageTextConfig,
    GlmImageVisionConfig,
)
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers


class GlmImageVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class GlmImageVisionAttention(nn.Module):
    def __init__(self, config: GlmImageVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(
            config.hidden_size, config.hidden_size * 3, bias=config.attention_bias
        )
        self.proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS["eager"]
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        if "flash" in self.config._attn_implementation:
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class GlmImageVisionPatchEmbed(nn.Module):
    def __init__(self, config: GlmImageVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.patch_size, self.patch_size]
        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, hidden_states) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class GlmImageVisionEmbeddings(nn.Module):
    def __init__(self, config: GlmImageVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.interpolated_method = "bilinear"

    def forward(
        self, embeddings, lengths, image_shapes, h_coords, w_coords
    ) -> torch.Tensor:
        """
        Forward pass with integrated position encoding adaptation using 2D interpolation.

        Args:
            embeddings: Input embeddings tensor
            lengths (torch.Tensor): Sequence lengths for each image in the batch.
            image_shapes (torch.Tensor): Tensor of shape [batch_size, 3] representing the image shapes (t, h, w).
            h_coords (torch.Tensor): Tensor of shape [total_seq] representing the h coordinate for each patch.
            w_coords (torch.Tensor): Tensor of shape [total_seq] representing the w coordinate for each patch.

        Returns:
            torch.Tensor: Embeddings with adapted position encoding added.
        """
        # Get position embedding parameters
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        device = pos_embed_weight.device

        # Convert inputs to tensors if needed
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=device, dtype=torch.long)

        # Prepare 2D position embedding
        orig_size_sq = pos_embed_weight.shape[0]
        orig_size = int(orig_size_sq**0.5)
        pos_embed_2d = (
            pos_embed_weight.view(orig_size, orig_size, hidden_size)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )

        # Calculate target dimensions for each patch
        target_h = torch.cat(
            [image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]
        ).to(device=device, dtype=torch.float32)
        target_w = torch.cat(
            [image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]
        ).to(device=device, dtype=torch.float32)

        # Normalize coordinates to [-1, 1] range for grid_sample
        norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
        norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

        # Create sampling grid
        grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

        # Perform bicubic interpolation
        interpolated_embed_fp32 = F.grid_sample(
            pos_embed_2d,
            grid,
            mode=self.interpolated_method,
            align_corners=False,
            padding_mode="border",
        )

        # Reshape and convert back to original dtype
        adapted_pos_embed_fp32 = (
            interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
        )
        adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(
            embeddings.device
        )

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed
        return embeddings


class GlmImageVisionBlock(nn.Module):
    def __init__(self, config: GlmImageVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GlmImageVisionAttention(config)
        self.mlp = GlmImageVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.Tensor` of shape `(num_images_or_videos + 1,)`):
            The cumulative sequence lengths of each image or video feature.
        position_embeddings (`tuple(torch.Tensor, torch.Tensor)` of shape `(num_patches, head_dim // 2)`):
            The cosine and sine position embeddings for vision attention.
        """
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)

#     # Keep half or full tensor for later concatenation
#     rotary_dim = cos.shape[-1]
#     q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
#     k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

#     # Apply rotary embeddings on the first half or full tensor
#     q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
#     k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

#     # Concatenate back to full shape
#     q_embed = torch.cat([q_embed, q_pass], dim=-1)
#     k_embed = torch.cat([k_embed, k_pass], dim=-1)
#     return q_embed, k_embed


class GlmImageTextMLP(nn.Module):
    def __init__(
        self,
        config: GlmImageTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class GlmImageTextAttention(nn.Module):
    """Multi-headed attention with RadixAttention for sglang."""

    def __init__(
        self,
        config: GlmImageTextConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", hidden_size // num_heads)

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = max(1, num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        rope_theta = config.rope_parameters.get("rope_theta", 10000)
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)

        # Build rope_scaling dict for M-RoPE support
        # GLM-Image uses mrope_section from rope_parameters
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is None and "mrope_section" in config.rope_parameters:
            rope_scaling = {
                "rope_type": config.rope_parameters.get("rope_type", "default"),
                "mrope_section": config.rope_parameters["mrope_section"],
            }

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=partial_rotary_factor,
            is_neox_style=False,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class GlmImageTextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GlmImageTextConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GlmImageTextAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = GlmImageTextMLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_self_attn_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)

        return hidden_states, residual


class GlmImageVQVAEVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embed_dim
        self.beta = getattr(config, "beta", 0.25)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
        hidden_state_flattened = hidden_state.view(-1, self.embedding_dim)

        # L2 normalize
        hidden_state = F.normalize(hidden_state, p=2, dim=-1)
        hidden_state_flattened = F.normalize(hidden_state_flattened, p=2, dim=-1)
        embedding = F.normalize(self.embedding.weight, p=2, dim=-1)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        distances = (
            torch.sum(hidden_state_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", hidden_state_flattened, embedding.transpose(0, 1)
            )
        )

        min_encoding_indices = torch.argmin(distances, dim=1)
        hidden_state_quant = embedding[min_encoding_indices].view(hidden_state.shape)

        # compute loss for embedding
        loss = torch.mean(
            (hidden_state_quant.detach() - hidden_state) ** 2
        ) + self.beta * torch.mean((hidden_state_quant - hidden_state.detach()) ** 2)

        # preserve gradients
        hidden_state_quant = hidden_state + (hidden_state_quant - hidden_state).detach()

        # reshape back to match original input shape
        hidden_state_quant = hidden_state_quant.permute(0, 3, 1, 2).contiguous()

        return hidden_state_quant, loss, min_encoding_indices


class GlmImageVQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quantize = GlmImageVQVAEVectorQuantizer(config)
        self.quant_conv = nn.Conv2d(config.latent_channels, config.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.latent_channels, 1)
        self.eval()  # GlmImage's VQ model is frozen

    def encode(self, hidden_states):
        hidden_states = self.quant_conv(hidden_states)
        quant, emb_loss, indices = self.quantize(hidden_states)
        return quant, emb_loss, indices


class GlmImageVisionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.embeddings = GlmImageVisionEmbeddings(config)
        self.patch_embed = GlmImageVisionPatchEmbed(config)

        head_dim = config.hidden_size // config.num_heads

        self.blocks = nn.ModuleList(
            [GlmImageVisionBlock(config) for _ in range(config.depth)]
        )

        self.head_dim = head_dim

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(pixel_values)
        image_type_ids = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_states = self.embeddings(
            hidden_states,
            seqlens,
            grid_thw,
            image_type_ids[:, 0].to(hidden_states.device),
            image_type_ids[:, 1].to(hidden_states.device),
        )

        # Transformer blocks (no position_embeddings needed, already added above)
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
            )
        return hidden_states


class GlmImageTextModel(nn.Module):
    def __init__(
        self,
        config: GlmImageTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: GlmImageTextDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class GlmImageModel(nn.Module):
    def __init__(
        self,
        config: GlmImageConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.visual = GlmImageVisionModel(config.vision_config)
        self.language_model = GlmImageTextModel(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.vqmodel = GlmImageVQVAE(config.vq_config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens


class GlmImageForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: GlmImageConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config

        self.model = GlmImageModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        # LM head
        if self.pp_group.is_last_rank:
            vision_vocab_size = getattr(config.text_config, "vision_vocab_size", 16512)
            self.vision_vocab_size = vision_vocab_size
            self.lm_head = ParallelLMHead(
                vision_vocab_size,
                config.text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        # Logits processor and pooler
        self.logits_processor = LogitsProcessor(config.text_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # MROPE enabled check
        rope_scaling = getattr(config.text_config, "rope_scaling", {}) or {}
        self.is_mrope_enabled = "mrope_section" in rope_scaling

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_embed_and_head(self):
        return self.model.language_model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.language_model.embed_tokens.weight
        self.model.language_model.embed_tokens.weight = embed
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = self.model.language_model.embed_tokens
        else:
            del self.lm_head.weight
            self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        visual_dtype = getattr(self.model.visual, "dtype", torch.float16)
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            visual_dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)

        assert (
            pixel_values.dim() == 2
        ), f"Expected 2D pixel_values, got {pixel_values.dim()}"
        assert (
            image_grid_thw.dim() == 2
        ), f"Expected 2D image_grid_thw, got {image_grid_thw.dim()}"

        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_image_tokens(
        self, hidden_states: torch.FloatTensor, image_grid_thw: torch.LongTensor = None
    ):
        """Tokenizes image features into discrete tokens with VQVAE module."""
        hidden_size = hidden_states.shape[-1]
        split_sizes = (image_grid_thw.prod(dim=-1)).tolist()
        hidden_states_list = torch.split(hidden_states, split_sizes, dim=0)

        all_image_toks = []
        for i, hs in enumerate(hidden_states_list):
            grid_t, grid_h, grid_w = image_grid_thw[i].tolist()
            hs = hs.view(grid_t, grid_h, grid_w, hidden_size)
            hs = hs.permute(0, 3, 1, 2).contiguous()
            _, _, image_toks = self.model.vqmodel.encode(hs)
            all_image_toks.append(image_toks)
        return torch.cat(all_image_toks, dim=0)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """Run forward pass for GLM-Image.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a batch.
            positions: Flattened (concatenated) position ids corresponding to a batch.
                **NOTE**: If mrope is enabled, the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,)`.
            forward_batch: Contains batch information including multimodal inputs.
            get_embedding: If True, return pooled embeddings instead of logits.
            pp_proxy_tensors: Proxy tensors for pipeline parallelism.
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model.language_model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # Note: GLM-Image uses merged gate_up_proj in HF checkpoint,
            # MergedColumnParallelLinear handles the sharding automatically
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Skip lm_head on non-last PP ranks
            if name.startswith("lm_head.") and not self.pp_group.is_last_rank:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = GlmImageForConditionalGeneration
