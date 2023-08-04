import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from scalers.scale_light.blocks.backbone import build_backbone_x
from scalers.scale_light.blocks.transformer_lite import build_lite_encoder
from scalers.scale_light.blocks.pos_embed import build_position_encoding
from scalers.scale_light.blocks.head import build_head
from scalers.scale_light.utils import PreprocessorX, get_qkv


class STARKScaleModel(nn.Module):
    """Modified from stark_s_plus_sp
    The goal is to achieve ultra-high speed (1000FPS)
    2021.06.24 We change the input datatype to standard Tensor, rather than NestedTensor
    2021.06.27 Definition of transformer is changed"""
    def __init__(self, backbone, transformer, head, pos_emb_x, pos_emb_z, params):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.head = head
        self.pos_emb_x = pos_emb_x
        self.pos_emb_z = pos_emb_z
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=1),
            nn.AvgPool2d(kernel_size=2)
            )  # the bottleneck layer
        self.processor = PreprocessorX()
        self.params = params


    def forward(self, img1: torch.tensor, img2: torch.tensor, mask1: torch.tensor, mask2: torch.tensor):

        x_dict = self.forward_backbone(img=img1, mask=mask1, zx='template')
        z_dict = self.forward_backbone(img=img2, mask=mask2, zx='search')
       # print(x_dict['feat'].shape)
       # print(x_dict['pos'].shape)
        feat_dict_list = [x_dict, z_dict]
        q, k, v, key_padding_mask = get_qkv(feat_dict_list)
        out_dict = self.forward_transformer(q=q, k=k, v=v, key_padding_mask=key_padding_mask)
        
        result = self.forward_head(out_dict)

        return result
    


    def forward_backbone(self, img: torch.Tensor, zx: str, mask: torch.Tensor):
        """The input type is standard tensor
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(img, torch.Tensor)
        """run the backbone"""
        output_back = self.backbone(img)  # features & masks, position embedding for the search
        """get the positional encoding"""
        bs = img.size(0)  # batch size
        if zx == "search":
            pos = self.pos_emb_x(bs)
        elif "template" in zx:
            pos = self.pos_emb_z(bs)
        else:
            raise ValueError("zx should be 'template_0' or 'search'.")
        """get the downsampled attention mask"""
        mask_down = F.interpolate(mask[None].float(), size=[20, 20]).to(torch.bool)[0]
        """adjust the shape"""
        return self.adjust(output_back, pos, mask_down)

    def forward_transformer(self, q, k, v, key_padding_mask=None, softmax=True):
        # run the transformer encoder
        enc_mem = self.transformer(q, k, v, key_padding_mask=key_padding_mask)
        return enc_mem

    def forward_head(self, memory, softmax=True):
        """ memory: encoder embeddings (HW1+HW2, B, C) / (HW2, B, C)"""
        
        # encoder output for the search region (H_x*W_x, B, C)
        memory = torch.permute(memory, (1, 0, 2))  # (B, C, H_x*W_x)
        output = self.head(memory)
        return output
        

    def adjust(self, src_feat: torch.Tensor, pos_embed: torch.Tensor, mask: torch.Tensor):
        """
        """
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}


def build_stark_scale_model(cfg, phase: str):
    """phase: 'train' or 'test'
    during the training phase, we need to
        (1) load backbone pretrained weights
        (2) freeze some layers' parameters"""
    backbone = build_backbone_x(cfg, phase=phase)
    for params in backbone.parameters():
        params.requires_grad = False
    transformer = build_lite_encoder(cfg)
    head = build_head(cfg)
    fsz_x, fsz_z = cfg.DATA.SEARCH.FEAT_SIZE, cfg.DATA.TEMPLATE.FEAT_SIZE
    pos_emb_x = build_position_encoding(cfg, fsz_x)
    pos_emb_z = build_position_encoding(cfg, fsz_z)
    model = STARKScaleModel(
        backbone,
        transformer,
        head,
        pos_emb_z,
        pos_emb_x,
        cfg,
    )

    return model

#from utils_model import read_config
#import cv2

#cfg_path = '/home/user/computer_vision/stark_scale/configs/scale.yaml'
#config = read_config(cfg_path)
#model = build_stark_scale_model(config, 'Train')

#print(model)
# process = PreprocessorX()

# img1 = cv2.imread('/home/user/Documents/arshak/image.jpg')
# img2 = cv2.imread('/home/user/Documents/arshak/image.jpg')

# img1_tensor, mask1_tensor = process.process(img1, 320)
# img2_tensor, mask2_tensor = process.process(img2, 320)

# model.eval()
# r = model(img1_tensor, img2_tensor, mask1_tensor, mask2_tensor)
# print(r.shape)

#def params_count_lite(model):
 #   total_params = sum(p.numel() for p in model.parameters())
  #  total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  #  print(f'Total parameters: {total_params}')
  #  print(f'Trainable parameters: {total_trainable_params}')

#print(params_count_lite(model))
