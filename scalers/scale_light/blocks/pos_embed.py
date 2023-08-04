"""
Various positional encodings for the transformer.
"""
import torch
from torch import nn

class PositionEmbedding(nn.Module):
    """
    Absolute pos embedding, learned. (allow users to specify the size)
    """
    def __init__(self, num_pos_feats=256, sz=20):
        super().__init__()
        self.sz = sz
        self.row_embed = nn.Embedding(sz, num_pos_feats)
        self.col_embed = nn.Embedding(sz, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, bs):
        """bs: batch size"""
        h, w = self.sz, self.sz
        i = torch.arange(w, device=self.col_embed.weight.device)
        j = torch.arange(h, device=self.row_embed.weight.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(bs, 1, 1, 1)
        return pos  # (H,W,C) --> (C,H,W) --> (1,C,H,W) --> (B,C,H,W)


def build_position_encoding(cfg, sz):
    N_steps = cfg.MODEL.HIDDEN_DIM // 2
    position_embedding = PositionEmbedding(N_steps, sz)
    return position_embedding



