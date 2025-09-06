import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .swin_transformer import build_swin_transformer

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {f"layer{i + 1}": str(idx) for idx, i in enumerate(return_interm_layers)}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_indices: list,
                 dilation: bool):
        if name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        assert name not in ('resnet18', 'resnet34'), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices):]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.return_interm_layers
    assert return_interm_layers in ([0,1,2,3], [1,2,3], [3])
    if args.backbone in ['resnet50', 'resnet101']:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        bb_num_channels = backbone.num_channels
    elif args.backbone in ['swin_T_224_22k', 'swin_B_384_22k', 'swin_L_384_22k']:
        pretrain_img_size = int(args.backbone.split('_')[-2])
        backbone = build_swin_transformer(args.backbone,
                                          pretrain_img_size=pretrain_img_size,
                                          out_indices=tuple(return_interm_layers),
                                          dilation=args.dilation, use_checkpoint=False)
        if args.backbone_pretrained:
            pretrained_dir = args.backbone_pretrained
            PTDICT = {
                'swin_T_224_22k': 'swin_tiny_patch4_window7_224_22k.pth',
                'swin_B_384_22k': 'swin_base_patch4_window12_384_22k.pth',
                'swin_L_384_22k': 'swin_large_patch4_window12_384_22k.pth',
            }
            pretrainedpath = os.path.join(pretrained_dir, PTDICT[args.backbone])
            checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
            from collections import OrderedDict
            def clean_state_dict(state_dict):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k[:7] == 'module.':
                        k = k[7:]  # remove `module.`
                    new_state_dict[k] = v
                return new_state_dict
            def key_select_function(keyname):
                if 'head' in keyname:
                    return False
                if args.dilation and 'layers.3' in keyname:
                    return False
                return True
            _tmp_st = OrderedDict({k: v for k, v in clean_state_dict(checkpoint).items() if key_select_function(k)})
            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))
        bb_num_channels = backbone.num_features[4 - len(return_interm_layers):]
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(return_interm_layers), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_layers)}"

    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(bb_num_channels, List), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    return model
