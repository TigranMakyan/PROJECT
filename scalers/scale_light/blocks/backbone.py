"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
import os, sys
sys.path.insert(0, '.')

import scalers.scale_light.blocks.resnet as resnet_module
from scalers.scale_light.blocks.repvgg import get_RepVGG_func_by_name

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

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
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    def __init__(self, name: str, dilation: bool, freeze_bn: bool, last_stage_block=14):
        super().__init__()
        if "resnet" in name:
            norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
            # here is different from the original DETR because we use feature from block3
            self.body = getattr(resnet_module, name)(
                replace_stride_with_dilation=[False, dilation, False],
                pretrained=True, norm_layer=norm_layer, last_layer='layer3')
            self.num_channels = 256 if name in ('resnet18', 'resnet34') else 1024
        elif "RepVGG" in name:
            print("#" * 10 + "  Warning: Dilation is not valid in current code  " + "#" * 10)
            repvgg_func = get_RepVGG_func_by_name(name)
            self.body = repvgg_func(deploy=False, last_layer="stage3", freeze_bn=freeze_bn,
                                    last_stage_block=last_stage_block)
            self.num_channels = 192  # 256x0.75=192
        else:
            raise ValueError("Unsupported net type")

    def forward(self, x: torch.Tensor):
        return self.body(x)


def build_backbone_x(cfg, phase='train'):
    """Without positional embedding, standard tensor input"""
    train_backbone = cfg.TRAIN.BACKBONE_MULTIPLIER > 0
    backbone = Backbone(cfg.MODEL.BACKBONE.TYPE, cfg.MODEL.BACKBONE.DILATION, cfg.TRAIN.FREEZE_BACKBONE_BN,
                        cfg.MODEL.BACKBONE.LAST_STAGE_BLOCK)

    if phase == 'train':
        """load pretrained backbone weights"""
        ckpt_path = None
        if hasattr(cfg, "ckpt_dir"):
            if cfg.MODEL.BACKBONE.TYPE == "RepVGG-A0":
                filename = "RepVGG-A0-train.pth"
            elif cfg.MODEL.BACKBONE.TYPE == "LightTrack":
                filename = "LightTrackM.pth"
            else:
                raise ValueError("The checkpoint file for backbone type %s is not found" % cfg.MODEL.BACKBONE.TYPE)
            ckpt_path = os.path.join(cfg.ckpt_dir, filename)
        if ckpt_path is not None:
            print("Loading pretrained backbone weights from %s" % ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if cfg.MODEL.BACKBONE.TYPE == "LightTrack":
                ckpt, ckpt_new = ckpt["state_dict"], {}
                for k, v in ckpt.items():
                    if k.startswith("features."):
                        k_new = k.replace("features.", "")
                        ckpt_new[k_new] = v
                ckpt = ckpt_new
            missing_keys, unexpected_keys = backbone.body.load_state_dict(ckpt, strict=False)
            
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)

        """freeze some layers"""
        if cfg.MODEL.BACKBONE.TYPE != "LightTrack":
            trained_layers = cfg.TRAIN.BACKBONE_TRAINED_LAYERS
            # defrost parameters of layers in trained_layers
            for name, parameter in backbone.body.named_parameters():
                parameter.requires_grad_(False)
                if train_backbone:
                    for trained_name in trained_layers:  # such as 'layer2' in layer2.conv1.weight
                        if trained_name in name:
                            parameter.requires_grad_(True)
                            break
    return backbone

# cfg_path = '/home/user/computer_vision/asenq/scalers/scale_light/configs/stark.yaml'
# from easydict import EasyDict
# import yaml

# def read_config(file_path):
#     with open(file_path, 'r') as file:
#         config_dict = yaml.safe_load(file)
#     config = EasyDict(config_dict)
#     return config
# config = read_config(cfg_path)
# backbone = build_backbone_x(config)

# # params_count_lite(backbone)
# a = torch.randn(1, 3, 512, 512)
# res = backbone(a)
# print(res.shape)


'''
Return BS, 1024, 64, 64 shape
resnet50
8512704 parameters
include dilation
'''


