import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from scalers.google.vgg_frontend import build_vgg
from scalers.google.linear import LinearModel
from scalers.google.dil_conv_block import build_dc


class RelScale(nn.Module):
    def __init__(self, freeze_bb=True) -> None:
        super(RelScale, self).__init__()
        self.bb = build_vgg(pretrained=True, freeze=freeze_bb)
        self.dc1 = build_dc()
        self.dc2 = build_dc()
        self.lin = LinearModel()

    def forward(self, img1, img2):
        img1 = img1.to(torch.float32)
        img2 = img2.to(torch.float32)

        x1 = self.bb(img1)
        x2 = self.bb(img2)
        
        x1 = torch.flatten(self.dc1(x1), start_dim=1)
        x2 = torch.flatten(self.dc2(x2), start_dim=1)
        
        input_catted = torch.cat((x1, x2), dim=1)
        result = self.lin(input_catted)
        return result   

    def compute_l1_loss(self, w):
      return torch.abs(w).sum()

def build_rel_scalenet(pretrained=False, freeze=False, freeze_bb=True):
    model = RelScale(freeze_bb=freeze_bb)
    if pretrained:
        checkpoint = torch.load('./models/model_best.pth')
        print(checkpoint['epoch'])
        model.load_state_dict(checkpoint['model_state_dict'])
    

    return model

    
model = build_rel_scalenet()



a = torch.rand(1, 3, 512, 512)
b = torch.rand(1, 3, 512, 512)

model = RelScale()
r = model(a, b)
print(r.shape)

