import torch
from torchvision.models import vgg16_bn
import torch.nn as nn

from dil_conv_block import initialize_weights

def build_vgg(pretrained=True, freeze=False):
    model = vgg16_bn()
    if pretrained:
        print('[INFO]: Loading VGG_16_bn weights')
        model.load_state_dict(torch.load('./weights/only_vgg.pth'))
        model = nn.Sequential(*list(model.features.children()))
    else:
        print('[INFO]: NOT Loading VGG_16_bn weights')
        model = nn.Sequential(*list(model.features.children()))
        initialize_weights(model)
    if freeze:
        print('[INFO]: FREEZE vgg frontend')
        for params in model.parameters():
            params.requires_grad = False
    else:
        print('[INFO]: BackPropogation via vgg frontend')
        for params in model.parameters():
            params.requires_grad = True
    return model


