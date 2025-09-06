import copy
import random

import torch
from typing import Optional, List
from torch import nn, Tensor
from util.misc import _get_clones
from .utils import deformable_attention_core_func, get_activation
import torch.nn.functional as F
import torch.nn.init as init
import math

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

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

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



class ATFmodule(nn.Module):
    def __init__(self):
        super(ATFmodule, self).__init__()
        self.attetion_block = nn.Sequential(nn.Linear(256*2,256),
                                            nn.ReLU(),
                                            nn.Linear(256,256),
                                            nn.Sigmoid())
        self.mlp_layer = nn.Sequential(nn.Linear(256*2,256),
                                       nn.ReLU(),
                                       nn.Linear(256,256),
                                       nn.LayerNorm(256))
    def forward(self,x,y):
        att = self.attetion_block(torch.cat([x,y],dim=-1))
        ret = x + att * self.mlp_layer(torch.cat([x,y],dim=-1))
        return ret

class ATF(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.args = args

        if self.args.sharing_fusion_module:
            self.atfm = ATFmodule()
        else:
            self.atfm = _get_clones(ATFmodule(),self.args.dec_layers)

    def forward(self,task_feat,context,n):
        if self.args.sharing_fusion_module:
            return self.atfm(task_feat,context)
        return self.atfm[n](task_feat,context)

def make_mlp(dim_in,dim_out):
    module = nn.Sequential(nn.Linear(dim_in,dim_out),
                           nn.ReLU(),
                           nn.Linear(dim_out,dim_out),
                           nn.LayerNorm(dim_out)
                           )
    return module

# 假设已导入或定义了可变形注意力模块
class ACF(nn.Module):
    def __init__(self, dim: int = 256, nhead: int = 8, feeddim: int = 2048, n_levels=3, n_points=4):
        super(ACF, self).__init__()
        self.dim = dim
        self.nhead = nhead
        self.feeddim = feeddim

        self.ternary = make_mlp(dim * 3, dim)

        self.human_obj = make_mlp(dim * 2, dim)
        self.human_rel = make_mlp(dim * 2, dim)
        self.obj_rel = make_mlp(dim * 2, dim)

        self.ternary_self = TransformerEncoderLayer(dim, nhead, dim_feedforward=feeddim)
        self.unary_self = TransformerEncoderLayer(dim, nhead, dim_feedforward=feeddim)
        self.pairwise_self = TransformerEncoderLayer(dim, nhead, dim_feedforward=feeddim)

        # self.human_unary = TransformerCrossLayer(dim, nhead, dim_feedforward=feeddim)
        # self.object_unary = TransformerCrossLayer(dim, nhead, dim_feedforward=feeddim)
        # self.verb_unary = TransformerCrossLayer(dim, nhead, dim_feedforward=feeddim)
        self.unary_cross = TransformerCrossLayer(dim, nhead, dim_feedforward=feeddim)
        self.pairwise_cross = TransformerCrossLayer(dim, nhead, dim_feedforward=feeddim)

        self.human_context = AcfLayer(dim, nhead, dim_feedforward=feeddim)
        self.object_context = AcfLayer(dim, nhead, dim_feedforward=feeddim)
        self.verb_context = AcfLayer(dim, nhead, dim_feedforward=feeddim)

        # 上下文生成层，用于与编码器记忆交互
        self.mc_gen = TransformerCrossLayer(dim, nhead,dim_feedforward=feeddim)

        # 可学习权重，用于调整上下文贡献
        self.weight_mlp_human = MLP(dim, dim // 2, 1, num_layers=2)
        self.weight_mlp_object = MLP(dim, dim // 2, 1, num_layers=2)
        self.weight_mlp_verb = MLP(dim, dim // 2, 1, num_layers=2)
    def forward(self, output_sub, output_obj, output_rel, decoding_stuff):
        """
        前向传播函数
        Args:
            output_sub: 人类查询，形状 (batch_size, num_queries, dim)
            output_obj: 物体查询，形状 (batch_size, num_queries, dim)
            output_rel: 动词查询，形状 (batch_size, num_queries, dim)
            decoding_stuff: 包含 memory, memory_key_padding_mask, pos 的字典
        Returns:
            更新后的人类、物体和动词查询，形状均为 (batch_size, num_queries, dim)
        """
        # 转置输入以匹配变换器期望的形状
        output_sub = output_sub.transpose(0, 1)  # (num_queries, batch_size, dim)
        output_obj = output_obj.transpose(0, 1)
        output_rel = output_rel.transpose(0, 1)

        # 提取解码器上下文
        output_memory, memory_key_padding_mask, pos_embeds = decoding_stuff
        outputs_ternary = self.ternary((torch.cat([output_sub, output_obj, output_rel], dim=-1)))
        outputs_ternary = self.ternary_self(outputs_ternary)

        # unary information encoding
        outputs_unary = self.unary_self(torch.stack([output_sub, output_obj, output_rel], dim=0).flatten(1, 2)) \
            .view(3, output_sub.size(0), output_sub.size(1), output_sub.size(2))
        human_unary, object_unary, verb_unary = outputs_unary[0], outputs_unary[1], outputs_unary[2]
        outputs_tu = self.unary_cross(tgt=outputs_ternary.flatten(0, 1).unsqueeze(0), memory=outputs_unary.flatten(1, 2)).view(output_sub.size(0), output_sub.size(1),
                                       output_sub.size(2))

        # second information encoding
        human_obj_feat = self.human_obj(torch.cat([outputs_unary[0], outputs_unary[1]], dim=-1))
        human_rel_feat = self.human_rel(torch.cat([outputs_unary[0], outputs_unary[2]], dim=-1))
        obj_rel_feat = self.obj_rel(torch.cat([outputs_unary[1], outputs_unary[2]], dim=-1))
        outputs_pairwise = self.pairwise_self(torch.stack([human_obj_feat, human_rel_feat, obj_rel_feat], dim=0).flatten(1, 2)) \
            .view(3, output_sub.size(0), output_sub.size(1), output_sub.size(2))
        outputs_tup = self.pairwise_cross(tgt=outputs_tu.flatten(0, 1).unsqueeze(0),memory=outputs_pairwise.flatten(1, 2)).view(output_sub.size(0),\
                                          output_sub.size(1), output_sub.size(2))

        multiplex_context = self.mc_gen(outputs_tup, output_memory.transpose(0,1), memory_key_padding_mask=memory_key_padding_mask.flatten(1,2),
                                        pos=pos_embeds.transpose(0,1))
        human_context = self.human_context(tgt=human_unary, memory=multiplex_context)
        object_context = self.object_context(tgt=object_unary, memory=multiplex_context)
        verb_context = self.verb_context(tgt=verb_unary, memory=multiplex_context)

        # w_sub = self.weight_mlp_human(human_context).softmax(dim=-1)  # [bs, num_queries, 3]
        # w_obj = self.weight_mlp_object(object_context).softmax(dim=-1)
        # w_rel = self.weight_mlp_verb(verb_context).softmax(dim=-1)

        return human_context.transpose(0, 1), object_context.transpose(0, 1), verb_context.transpose(0, 1)

class TransformerDeformableLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, n_levels=3, n_points=4,
                 activation="relu", normalize_before=False,return_attn=False,kdim=None,vdim=None):
        super().__init__()
        self.deformable_attn = MSDeformableAttention(d_model, nhead, n_levels, n_points)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation(activation)
        self.normalize_before = normalize_before
        self.return_attn = return_attn

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, reference_points, memory, spatial_shapes, memory_mask=None):

        tgt2 = self.deformable_attn(tgt, reference_points, memory, spatial_shapes, value_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class AcfLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,return_attn=False,kdim=None,vdim=None):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,kdim=kdim,vdim=vdim)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation(activation)
        self.normalize_before = normalize_before
        self.return_attn = return_attn
        self.weight1 = MLP(d_model, d_model // 2, 1, num_layers=2)
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        tgt2, cross_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        w1 = self.weight1(tgt2).softmax(dim=-1)
        tgt = tgt + w1 * self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class TransformerCrossLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,return_attn=False,kdim=None,vdim=None):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,kdim=kdim,vdim=vdim)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation(activation)
        self.normalize_before = normalize_before
        self.return_attn = return_attn
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        tgt2, cross_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt