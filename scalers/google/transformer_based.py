import torch
import torch.nn as nn

from models.vgg_frontend import build_vgg
from models.linear import LinearModel
from models.dil_conv_block import build_dc
from models.utils import initialize_weights

class RelScaleTransformer(nn.Module):
    def __init__(self, pretrained=False, freeze_bb=True) -> None:
        super(RelScaleTransformer, self).__init__()
        self.bb = build_vgg(pretrained=True, freeze=freeze_bb)
        self.dc1 = build_dc()
        self.dc2 = build_dc()
        self.transformer = nn.Transformer(d_model=1024, num_encoder_layers=2, num_decoder_layers=2, nhead=8)
        self.lin = LinearModel()
        initialize_weights(self.dc1)
        initialize_weights(self.dc2)
        initialize_weights(self.transformer)
        initialize_weights(self.lin)

    def forward(self, img1, img2):
        img1 = img1.to(torch.float32)
        img2 = img2.to(torch.float32)

        x1 = self.bb(img1)
        x2 = self.bb(img2)
       
        x1 = torch.flatten(self.dc1(x1), start_dim=1)
        x2 = torch.flatten(self.dc2(x2), start_dim=1)
       
        input_catted = torch.cat((x1, x2), dim=1)

        input_catted = input_catted.view(-1, 1, 1024)

        out = self.transformer(input_catted, input_catted)
        out = out.view(-1, 8192)

        out = torch.flatten(out, start_dim=1)
        out = self.lin(out)
        return out   


def build_relscaletransformer(pretrained=False):
    model = RelScaleTransformer()
    if pretrained:
        checkpoint = torch.load('./models/model_adam_17_2606.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

#a = torch.rand(4, 3, 512, 512)
#b = torch.rand(4, 3, 512, 512)

#model = RelScaleTransformer()
#r = model(a, b)
#print(r.shape)
#params_count_lite(model)
#print(b.shape)
#params_count_lite(model.transformer)
#params_count_lite(model.dc1)
#params_count_lite(model.lin)
