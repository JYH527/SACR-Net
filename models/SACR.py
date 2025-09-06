from pathlib import Path

import numpy as np
import torchvision
from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F
import copy
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_xyxy_to_cxcywh, box_iou,masks_to_boxes, box_area
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .hybrid_encoder import HybridEncoder
from .decoder import RTDETRTransformer
from .backbone import build_backbone
from .matcher import build_matcher

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class SACR(nn.Module):
    def __init__(self, backbone, nhead, hidden_dim, dim_forward, num_encoder_layers, num_obj_classes, num_verb_classes, num_queries, aux_loss,
                 num_levels, num_decoder_layers, use_matching, fusion_query=False, eval_spatial_size=None):
        super().__init__()
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        self.num_obj_classes = num_obj_classes
        self.num_verb_classes = num_verb_classes
        self.num_queries = num_queries
        self.dec_layers = num_decoder_layers
        self.use_matching = use_matching
        self.matching_embed = MLP(hidden_dim*2, hidden_dim, 2, 3)
        self.eval_spatial_size = eval_spatial_size
        self.encoder = HybridEncoder(nhead, hidden_dim, dim_forward, num_encoder_layers, in_channels=[512, 1024, 2048],eval_spatial_size=eval_spatial_size)
        self.decoder = RTDETRTransformer(nhead, hidden_dim, num_queries, num_levels, num_decoder_layers, dim_forward, num_obj_classes, num_verb_classes,fusion_query=fusion_query,
                                         eval_spatial_size=eval_spatial_size)

    def forward(self, samples: NestedTensor):
        # samples = NestedTensor(samples,samples2)
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        srcs = []
        masks = []
        for i, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert masks is not None
        outs = self.encoder(srcs)
        human_output, object_output, verb_output, outputs_human_coords, outputs_object_classes, outputs_object_coords, outputs_verb_classes,\
            enc_verb_class, enc_human_reference_points, enc_object_class, enc_object_reference_points = self.decoder(outs)
        nT = nn.Tanh()
        if self.training:
            out = {'pred_obj_logits': outputs_object_classes[-1] * 1 - nT(outputs_object_classes.var(dim=0)),
                   'pred_verb_logits': outputs_verb_classes[-1] * 1 - nT(outputs_verb_classes.var(dim=0)),
                'pred_sub_boxes': outputs_human_coords[-1], 'pred_obj_boxes': outputs_object_coords[-1],
                'sub_out': human_output, 'obj_out': object_output, 'rel_out': verb_output}
            if self.use_matching:
                outputs_matching = self.matching_embed(torch.cat([human_output, object_output], dim=-1))
                out['pred_matching_logits'] = outputs_matching[-1] * 1 - nT(outputs_matching.var(dim=0))
        else:
            out = {'pred_obj_logits': outputs_object_classes[-1], 'pred_verb_logits': outputs_verb_classes[-1],
                   'pred_sub_boxes': outputs_human_coords[-1], 'pred_obj_boxes': outputs_object_coords[-1],
                   'sub_out': human_output, 'obj_out': object_output, 'rel_out': verb_output}
            if self.use_matching:
                outputs_matching = self.matching_embed(torch.cat([human_output, object_output], dim=-1))
                out['pred_matching_logits'] = outputs_matching[-1]
        out["enc_outputs"] = {'pred_obj_logits': enc_object_class, 'pred_verb_logits': enc_verb_class,
                               'pred_sub_boxes': enc_human_reference_points, 'pred_obj_boxes': enc_object_reference_points}
        if self.aux_loss:
            if self.use_matching:
                out['aux_outputs'] = self._set_aux_loss(outputs_object_classes, outputs_verb_classes,
                                                        outputs_human_coords, outputs_object_coords, outputs_matching)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_object_classes, outputs_verb_classes,
                                                        outputs_human_coords, outputs_object_coords)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_matching=None):
        min_dec_layers_num = self.dec_layers
        if self.use_matching:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d, 'pred_matching_logits': e}
                    for a, b, c, d, e in zip(outputs_obj_class[-min_dec_layers_num : -1], outputs_verb_class[-min_dec_layers_num : -1],
                                        outputs_sub_coord[-min_dec_layers_num : -1], outputs_obj_coord[-min_dec_layers_num : -1], outputs_matching[-min_dec_layers_num : -1])]
        else:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[-min_dec_layers_num: -1], outputs_verb_class[-min_dec_layers_num: -1],
                                        outputs_sub_coord[-min_dec_layers_num: -1], outputs_obj_coord[-min_dec_layers_num: -1])]

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


class SetCriterionHOI(nn.Module):
    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()
        self.args = args
        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.alpha = args.alpha

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        obj_weights = self.empty_weight
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, obj_weights)
        losses = {'loss_obj_ce': loss_obj_ce}
        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_verb_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_verb_logits']
        device = pred_logits.device
        # 计算每个样本的真实动词标签数量
        tgt_lengths = torch.as_tensor([v['verb_labels'].shape[0] for v in targets], device=device, dtype=torch.long)
        # 预测的动词标签数量（非背景类的数量）
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # 计算基数误差（L1损失）
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'loss_verb_cardinality': card_err}
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'loss_obj_cardinality': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        # 提取模型预测的动词 logits（假设形状为 [batch_size, num_queries, 29]）
        src_logits = outputs['pred_verb_logits']
        # 获取匹配预测的索引
        idx = self._get_src_permutation_idx(indices)
        # 从 targets 中收集 one-hot 编码的动词标签（形状 [6, 29]）
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        # 验证形状兼容性
        assert target_classes_o.shape[1] == self.num_verb_classes, "动词标签的类别数必须匹配"
        # 初始化目标张量，形状与 src_logits 一致，填充为零
        target_classes = torch.zeros_like(src_logits)
        # 将 one-hot 标签赋值到对应的索引位置
        if target_classes_o.shape[0] > 0:
            target_classes[idx] = target_classes_o  # 形状匹配：[6, 29] 到 [6, 29]

        # 对 logits 应用 sigmoid 激活，得到多标签概率
        src_logits = src_logits.sigmoid()

        # 可选：计算类别权重以处理数据不平衡
        class_counts = target_classes_o.sum(dim=0).float()
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * self.num_verb_classes

        # 计算二元交叉熵损失
        loss_verb_ce = F.binary_cross_entropy(
            src_logits, target_classes, weight=class_weights[None, None, :], reduction='mean'
        )

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses


    def loss_matching_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_matching_logits' in outputs
        src_logits = outputs['pred_matching_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['matching_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_matching = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_matching': loss_matching}
        if log:
            losses['matching_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _neg_loss(self, pred, gt, alpha=0.25):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        loss = 0
        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'verb_cardinality': self.loss_verb_cardinality,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'matching_labels': self.loss_matching_labels,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets, enc_outputs=False)
        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # 应用权重
        weighted_losses = {}
        for loss_name, loss_value in losses.items():
            if loss_name in self.weight_dict:
                weighted_losses[loss_name] = loss_value * self.weight_dict[loss_name]
            else:
                weighted_losses[loss_name] = loss_value  # 未指定权重的损失保持原值

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            required_keys = ['pred_obj_logits', 'pred_verb_logits', 'pred_sub_boxes', 'pred_obj_boxes']
            if not all(key in enc_outputs for key in required_keys):
                raise ValueError(f"enc_outputs missing required keys: {required_keys}")
            indices = self.matcher(enc_outputs, targets, enc_outputs=True)
            enc_losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality', 'verb_cardinality']
            for loss in enc_losses:
                kwargs = {'log': False} if loss == 'obj_labels' else {}
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_interactions, **kwargs)
                l_dict = {k + f'_enc': v * self.weight_dict.get(k, 1.0) for k, v in l_dict.items()}
                weighted_losses.update(l_dict)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, enc_outputs=False)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v * self.weight_dict.get(k + f'_{i}', 1.0) for k, v in l_dict.items()}
                    weighted_losses.update(l_dict)

        return weighted_losses


class PostProcessHOI(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id
        self.use_matching = args.use_matching

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits = outputs['pred_obj_logits']
        out_verb_logits = outputs['pred_verb_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        if self.use_matching:
            out_matching_logits = outputs['pred_matching_logits']
            matching_scores = F.softmax(out_matching_logits, -1)[..., 1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(obj_scores)):
            os, ol, vs, sb, ob =  obj_scores[index], obj_labels[index], verb_scores[index], sub_boxes[index], obj_boxes[index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            if self.use_matching:
                ms = matching_scores[index]
                vs = vs * ms.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:],'obj_scores':os.to('cpu')})

        return results

def build(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)

    model = SACR(
        backbone,
        nhead=args.nheads,
        dim_forward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        num_queries=args.num_queries,
        num_levels=args.num_levels,
        hidden_dim=args.hidden_dim,
        num_obj_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
        aux_loss=args.aux_loss,
        use_matching=args.use_matching,
        fusion_query=args.fusion_query
    )

    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    weight_dict['loss_verb_cardinality'] = args.verb_cardinality_loss_coef
    weight_dict['loss_obj_cardinality'] = args.obj_cardinality_loss_coef
    if args.use_matching:
        weight_dict['loss_matching'] = args.matching_loss_coef

    if args.aux_loss:
        min_dec_layers_num = args.dec_layers
        aux_weight_dict = {}
        for i in range(min_dec_layers_num - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
    if args.use_matching:
        losses.append('matching_labels')


    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                            weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                            args=args)

    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOI(args)}

    return model, criterion, postprocessors
