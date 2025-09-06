"""by lyuwenyu
"""

import math 
import copy 
from collections import OrderedDict

from timm.layers import DropPath

from .acf import ACF
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob, gen_sineembed_for_position


__all__ = ['RTDETRTransformer']



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SAQBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(self, num_queries, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.1, attn_drop=0., drop_path=0.1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, tokens_norm=False):
        super().__init__()
        self.num_queries = num_queries
        self.norm1 = norm_layer(dim)

        self.attn = SAQAttn(num_queries,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.gamma1 = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(dim), requires_grad=True)

        # See https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721
        self.tokens_norm = tokens_norm

    def forward(self, x):
        x_norm1 = self.norm1(x)
        x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, self.num_queries:]], dim=1)
        x = x + self.drop_path(self.gamma1 * x_attn)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = torch.cat([self.norm2(x[:, :self.num_queries]), x[:, self.num_queries:]], dim=1)
        x_res = x
        det_token = x[:, :self.num_queries]
        det_token = self.gamma2 * self.mlp(det_token)
        x = torch.cat([det_token, x[:, self.num_queries:]], dim=1)
        x = x_res + self.drop_path(x)
        return x

class SAQAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, num_queries, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, :self.num_queries]).unsqueeze(1).reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_det = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_det = self.proj(x_det)
        x_det = self.proj_drop(x_det)

        return x_det

class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=3, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1]
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding
        Returns:
            output (Tensor): [bs, query_length, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        # Adjust reference_points if num_levels mismatch
        if reference_points.shape[-2] != self.num_levels:
            if reference_points.shape[-2] == 1:
                reference_points = reference_points.expand(-1, -1, self.num_levels, -1)
            else:
                raise ValueError(
                    f"reference_points num_levels ({reference_points.shape[-2]}) does not match "
                    f"self.num_levels ({self.num_levels}), got shape {reference_points.shape}"
                )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes, device=query.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                    reference_points[:, :, None, :, None, :2] + sampling_offsets /
                    self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=3,
                 n_points=4,):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # self._reset_parameters()

    # def _reset_parameters(self):
    #     linear_init_(self.linear1)
    #     linear_init_(self.linear2)
    #     xavier_uniform_(self.linear1.weight)
    #     xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        # if attn_mask is not None:
        #     attn_mask = torch.where(
        #         attn_mask.to(torch.bool),
        #         torch.zeros_like(attn_mask),
        #         torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes,
            memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, modulate_hw_attn=True, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.modulate_hw_attn = modulate_hw_attn
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.ACF = ACF()
        self.verb_query_scale = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.ref_point_head_human = MLP(hidden_dim, hidden_dim, 2, 2)
        self.ref_point_head_object = MLP(hidden_dim, hidden_dim, 2, 2)
    def forward(self, human_tgt, human_init_reference_points, object_tgt, object_init_reference_points, verb_tgt, verb_init_reference_points,
                output_memory, spatial_shapes, level_start_index, pos_embeds, valid_mask, bbox_head_human, bbox_head_object, bbox_head_rel,
                query_pos_head_human, query_pos_head_object, query_pos_head_rel, attn_mask=None, memory_mask=None):
        human_output = human_tgt
        object_output = object_tgt
        verb_output = verb_tgt
        human_output_intermedia = []
        object_output_intermedia = []
        verb_output_intermedia = []
        human_intermediate_reference_points = []
        object_intermediate_reference_points = []
        verb_intermediate_reference_points = []

        for i, layer in enumerate(self.layers):
            # human
            human_center = human_init_reference_points[..., :2]
            human_ref_points_input = human_init_reference_points.unsqueeze(2)
            query_sine_embed_human = gen_sineembed_for_position(human_ref_points_input[:, :, 0, :])
            query_pos_embed_human = query_pos_head_human(query_sine_embed_human)
            if self.modulate_hw_attn:
                refHW_cond = self.ref_point_head_human(human_output).sigmoid()  # nq, bs, 2
                query_pos_embed_human[..., self.hidden_dim // 2:] *= (refHW_cond[..., 0] / human_center[..., 0]).unsqueeze(-1)
                query_pos_embed_human[..., :self.hidden_dim // 2] *= (refHW_cond[..., 1] / human_center[..., 1]).unsqueeze(-1)
            human_output = layer(human_output, human_ref_points_input, output_memory, spatial_shapes, level_start_index,
                           attn_mask, memory_mask, query_pos_embed_human)

            # object
            object_center = object_init_reference_points[..., :2]
            object_ref_points_input = object_init_reference_points.unsqueeze(2)
            query_sine_embed_object = gen_sineembed_for_position(object_ref_points_input[:, :, 0, :])
            query_pos_embed_object = query_pos_head_object(query_sine_embed_object)
            if self.modulate_hw_attn:
                refHW_cond = self.ref_point_head_object(object_output).sigmoid()  # nq, bs, 2
                query_pos_embed_object[..., self.hidden_dim // 2:] *= (refHW_cond[..., 0] / object_center[..., 0]).unsqueeze(-1)
                query_pos_embed_object[..., :self.hidden_dim // 2] *= (refHW_cond[..., 1] / object_center[..., 1]).unsqueeze(-1)
            object_output = layer(object_output, object_ref_points_input, output_memory, spatial_shapes, level_start_index,
                                  attn_mask, memory_mask, query_pos_embed_object)
            # rel
            verb_ref_points_input = verb_init_reference_points.unsqueeze(2)
            query_sine_embed_verb = gen_sineembed_for_position(verb_ref_points_input[:, :, 0, :])
            query_pos_embed_verb = query_pos_head_rel(query_sine_embed_verb)
            query_scale_verb = self.verb_query_scale(verb_output) if i != 0 else 1
            query_pos_embed_verb = query_scale_verb * query_pos_embed_verb
            verb_output = layer(verb_output, verb_ref_points_input, output_memory,
                                spatial_shapes, level_start_index,
                                attn_mask, memory_mask, query_pos_embed_verb)

            # fusion
            human_output, object_output, verb_output = self.ACF(human_output, object_output, verb_output,
                                                                 (output_memory, valid_mask, pos_embeds))

            # iteration
            # iteration-human
            if bbox_head_human is not None:
                tmp = bbox_head_human[i](human_output)
                new_human_reference_points = tmp + inverse_sigmoid(human_init_reference_points)
                new_human_reference_points = new_human_reference_points.sigmoid()

                human_init_reference_points = new_human_reference_points
            human_intermediate_reference_points.append(human_init_reference_points)

            # iteration
            # iteration-object
            if bbox_head_object is not None:
                tmp = bbox_head_object[i](human_output)
                new_object_reference_points = tmp + inverse_sigmoid(object_init_reference_points)
                new_object_reference_points = new_object_reference_points.sigmoid()

                object_init_reference_points = new_object_reference_points
            object_intermediate_reference_points.append(object_init_reference_points)

            # iteration
            # iteration-rel
            if bbox_head_rel is not None:
                tmp = bbox_head_rel[i](verb_output)
                new_verb_reference_points = tmp + inverse_sigmoid(verb_init_reference_points)
                new_verb_reference_points = new_verb_reference_points.sigmoid()

                verb_init_reference_points = new_verb_reference_points
            verb_intermediate_reference_points.append(verb_init_reference_points)

            human_output_intermedia.append(human_output)
            object_output_intermedia.append(object_output)
            verb_output_intermedia.append(verb_output)

        return torch.stack(human_output_intermedia), torch.stack(human_intermediate_reference_points), \
               torch.stack(object_output_intermedia), torch.stack(object_intermediate_reference_points), \
               torch.stack(verb_output_intermedia), torch.stack(verb_intermediate_reference_points)

class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']
    def __init__(self,
                 nhead=8,
                 hidden_dim=256,
                 num_queries=100,
                 num_levels=3,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 num_object_classes=81,
                 num_verb_classes=29,
                 position_embed_type='sine',
                 feat_channels=[256, 256, 256],
                 feat_strides=[8, 16, 32],
                 num_decoder_points=4,
                 dropout=0.1,
                 activation="relu",
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True,
                 fusion_query=False):

        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_object_classes = num_object_classes
        self.num_verb_classes = num_verb_classes
        self.num_queries = num_queries
        self.fusion_query = fusion_query
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.query_embed_human = nn.Embedding(num_queries, hidden_dim * 2).weight
        self.query_embed_object = nn.Embedding(num_queries, hidden_dim * 2).weight
        self.query_embed_rel = nn.Embedding(num_queries, hidden_dim * 2).weight
        self.eval_idx = eval_idx if eval_idx >= 0 else num_decoder_layers + eval_idx
        # backbone feature projection
        self._build_input_proj_layer(feat_channels)
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, )
        )
        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.human_attn_blocks = nn.Sequential(*nn.ModuleList([
            SAQBlock(num_queries, dim=hidden_dim, num_heads=8, tokens_norm=True)
            for _ in range(2)]))

        self.object_attn_blocks = nn.Sequential(*nn.ModuleList([
            SAQBlock(num_queries, dim=hidden_dim, num_heads=8, tokens_norm=True)
            for _ in range(2)]))

        self.verb_attn_blocks = nn.Sequential(*nn.ModuleList([
            SAQBlock(num_queries, dim=hidden_dim, num_heads=8, tokens_norm=True)
            for _ in range(2)]))

        self.human_attn_blocks_pos = nn.Sequential(*nn.ModuleList([
            SAQBlock(num_queries, dim=hidden_dim, num_heads=8, tokens_norm=True)
            for _ in range(2)]))

        self.object_attn_blocks_pos = nn.Sequential(*nn.ModuleList([
            SAQBlock(num_queries, dim=hidden_dim, num_heads=8, tokens_norm=True)
            for _ in range(2)]))

        self.verb_attn_blocks_pos = nn.Sequential(*nn.ModuleList([
            SAQBlock(num_queries, dim=hidden_dim, num_heads=8, tokens_norm=True)
            for _ in range(2)]))

        # decoder embedding
        self.query_pos_head_human = MLP(hidden_dim * 2, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.query_pos_head_object = MLP(hidden_dim * 2, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.query_pos_head_rel = MLP(hidden_dim, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.bbox_embed_human = MLP(hidden_dim, hidden_dim, 4, num_layers=2)
        self.bbox_embed_object = MLP(hidden_dim, hidden_dim, 4, num_layers=2)
        self.verb_point_embed = MLP(hidden_dim, hidden_dim, 2, num_layers=2)
        self.verb_class_enc = nn.Linear(hidden_dim, num_verb_classes)
        self.object_class_enc = nn.Linear(hidden_dim, num_object_classes + 1)

        # decoder head
        self.dec_score_head_object = nn.ModuleList([
            nn.Linear(hidden_dim, num_object_classes + 1)
            for _ in range(num_decoder_layers)
        ])
        self.dec_score_head_rel = nn.ModuleList([
            nn.Linear(hidden_dim, num_verb_classes)
            for _ in range(num_decoder_layers)
        ])

        self.dec_bbox_head_human = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head_object = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head_rel = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 2, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()
            
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for reg_ in self.dec_bbox_head_human:
            init.constant_(reg_.layers[-1].weight.data, 0)
            init.constant_(reg_.layers[-1].bias.data[:2], 0.5)
            init.constant_(reg_.layers[-1].bias.data[2:], -2.0)
        for cls_, reg_ in zip(self.dec_score_head_rel, self.dec_bbox_head_rel):
            init.constant_(cls_.bias.data, bias_value)
            init.constant_(reg_.layers[-1].weight.data, 0)
            init.constant_(reg_.layers[-1].bias.data, 0.5)
        for cls_, reg_ in zip(self.dec_score_head_object, self.dec_bbox_head_object):
            init.constant_(cls_.bias.data, bias_value)
            init.constant_(reg_.layers[-1].weight.data, 0)
            init.constant_(reg_.layers[-1].bias.data[:2], 0.5)
            init.constant_(reg_.layers[-1].bias.data[2:], -2.0)
        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.query_pos_head_human.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head_human.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head_object.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head_object.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head_rel.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head_rel.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )
        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        pos_embeds = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            bs, c, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])
            pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, 10000).to(feat.device)
            pos_embeds.append(pos_embed)
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        pos_embeds = torch.cat(pos_embeds, dim=1).expand(bs, -1, -1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index, pos_embeds)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes):
        bs, _, c = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export
        valid_mask = valid_mask.expand(bs, -1, -1)
        query_embed_human = self.query_embed_human.unsqueeze(0).expand(bs, -1, -1) #(2,100,512)
        query_embed_object = self.query_embed_object.unsqueeze(0).expand(bs, -1, -1) #(2,100,512)
        query_embed_verb = self.query_embed_rel.unsqueeze(0).expand(bs, -1, -1) #(2,100,512)

        if self.fusion_query:
            output_memory = self.enc_output(memory)
            fusion_shape = spatial_shapes[-1]
            index = fusion_shape[0] * fusion_shape[1]
            query_embed_human, human_tgt = torch.split(query_embed_human, c, dim=2)
            query_embed_object, object_tgt = torch.split(query_embed_object, c, dim=2)
            query_embed_verb, verb_tgt = torch.split(query_embed_verb, c, dim=2)

            human_token = torch.cat([human_tgt, memory[:, -index:, :]], dim=1)
            obj_token = torch.cat([object_tgt, memory[:, -index:, :]], dim=1)
            verb_token = torch.cat([verb_tgt, memory[:, -index:, :]], dim=1)
            human_tgt = self.human_attn_blocks(human_token)[:, :self.num_queries]
            object_tgt = self.object_attn_blocks(obj_token)[:, :self.num_queries]
            verb_tgt = self.verb_attn_blocks(verb_token)[:, :self.num_queries]

            query_embed_human = torch.cat([query_embed_human, memory], dim=1)
            query_embed_object = torch.cat([query_embed_object, memory], dim=1)
            query_embed_verb = torch.cat([query_embed_verb, memory], dim=1)
            query_embed_human = self.human_attn_blocks_pos(query_embed_human)[:, :self.num_queries]
            query_embed_object = self.object_attn_blocks_pos(query_embed_object)[:, :self.num_queries]
            query_embed_verb = self.verb_attn_blocks_pos(query_embed_verb)[:, :self.num_queries]
        else:
            output_memory = self.enc_output(memory)
            query_embed_human, human_tgt = torch.split(query_embed_human, c, dim=1)
            query_embed_object, object_tgt = torch.split(query_embed_object, c, dim=1)
            query_embed_verb, verb_tgt = torch.split(query_embed_verb, c, dim=1)

        verb_init_reference_points = self.verb_point_embed(query_embed_verb).sigmoid()
        enc_verb_class = self.verb_class_enc(verb_tgt)

        enc_human_reference_points = self.bbox_embed_human(query_embed_human).sigmoid()
        human_init_reference_points = enc_human_reference_points

        enc_object_init_reference_points = self.bbox_embed_object(query_embed_object).sigmoid()
        object_init_reference_points = enc_object_init_reference_points
        enc_object_class = self.object_class_enc(object_tgt)

        return output_memory, human_tgt, human_init_reference_points, object_tgt, object_init_reference_points, verb_tgt, verb_init_reference_points,\
               enc_verb_class, enc_human_reference_points, enc_object_class, enc_object_init_reference_points, valid_mask


    def forward(self, feats, targets=None):

        # input projection and embedding
        (memory, spatial_shapes, level_start_index, pos_embeds) = self._get_encoder_input(feats)

        output_memory, human_tgt, human_init_reference_points, object_tgt, object_init_reference_points, verb_tgt, verb_init_reference_points,\
        enc_verb_class, enc_human_reference_points, enc_object_class, enc_object_reference_points, valid_mask = self._get_decoder_input(memory, spatial_shapes)

        human_out_boxs = []
        object_out_logits = []
        object_out_boxs = []
        verb_out_logits = []
        # decoder
        human_output, human_reference_points, object_output, object_reference_points, verb_output, verb_reference_points = self.decoder(human_tgt, human_init_reference_points, object_tgt, object_init_reference_points, verb_tgt, verb_init_reference_points,
                                            output_memory, spatial_shapes, level_start_index, pos_embeds, valid_mask,
                                            self.dec_bbox_head_human, self.dec_bbox_head_object, self.dec_bbox_head_rel,
                                            self.query_pos_head_human, self.query_pos_head_object, self.query_pos_head_rel)

        for lvl in range(human_output.shape[0]):
            if lvl == 0:
                human_reference = human_init_reference_points
            else:
                human_reference = human_reference_points[lvl - 1]
            human_reference = inverse_sigmoid(human_reference)
            tmp = self.dec_bbox_head_human[lvl](human_output[lvl])
            if human_reference.shape[-1] == 4:
                tmp += human_reference
            else:
                assert human_reference.shape[-1] == 2
                tmp[..., :2] += human_reference
            human_outputs_coord = tmp.sigmoid()
            human_out_boxs.append(human_outputs_coord)
        human_out_boxs = torch.stack(human_out_boxs)

        for lvl in range(object_output.shape[0]):
            if lvl == 0:
                object_reference = object_init_reference_points
            else:
                object_reference = object_reference_points[lvl - 1]
            object_reference = inverse_sigmoid(object_reference)
            object_outputs_class = self.dec_score_head_object[lvl](object_output[lvl])
            tmp = self.dec_bbox_head_object[lvl](object_output[lvl])
            if object_reference.shape[-1] == 4:
                tmp += object_reference
            else:
                assert object_reference.shape[-1] == 2
                tmp[..., :2] += object_reference
            object_outputs_coord = tmp.sigmoid()
            object_out_logits.append(object_outputs_class)
            object_out_boxs.append(object_outputs_coord)
        object_out_logits = torch.stack(object_out_logits)
        object_out_boxs = torch.stack(object_out_boxs)

        for lvl in range(verb_output.shape[0]):
            if lvl == 0:
                verb_reference = verb_init_reference_points
            else:
                verb_reference = verb_reference_points[lvl - 1]
            verb_reference = inverse_sigmoid(verb_reference)
            verb_outputs_class = self.dec_score_head_rel[lvl](verb_output[lvl])
            tmp = self.dec_bbox_head_rel[lvl](verb_output[lvl])
            if verb_reference.shape[-1] == 4:
                tmp += verb_reference
            else:
                assert verb_reference.shape[-1] == 2
                tmp[..., :2] += verb_reference
            verb_out_logits.append(verb_outputs_class)
        verb_out_logits = torch.stack(verb_out_logits)

        return human_output, object_output, verb_output, human_out_boxs, object_out_logits, object_out_boxs, verb_out_logits,\
               enc_verb_class, enc_human_reference_points, enc_object_class, enc_object_reference_points

