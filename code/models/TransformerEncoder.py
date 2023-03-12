import torch
from torch import nn
from torch import Tensor
import copy

from models.Attention import MultiHeadAttention


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 归一化层
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 残差链接层
class ResidualAdd(nn.Module):

    def __init__(self, size, dropout, rezero):
        super(ResidualAdd, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# 前馈全连接层
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, expansion: int = 4, dropout: float = 0.1):
        super(FeedForwardBlock, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * expansion)
        self.w_2 = nn.Linear(d_model * expansion, d_model)
        self.dropout = nn.Dropout(dropout)
        self.GELU = nn.GELU()

    def forward(self, x):
        return self.dropout(self.w_2(self.dropout(self.GELU(self.w_1(x)))))


# 单个Encoder层
class EncoderLayer(nn.Module):

    def __init__(self, size: int, self_attn, feed_forward, dropout: float = 0.1, rezero=True):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualAdd(size, dropout, rezero), 2)
        self.size = size

    def forward(self, x, mask: Tensor = None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, emb_size: int = 768, depth: int = 12):
        super(Encoder, self).__init__()
        self.depth = depth
        self.dropout = 0.0
        self.encodelayer = EncoderLayer(emb_size, MultiHeadAttention(emb_size, 6),
                                        FeedForwardBlock(emb_size, 4, dropout=self.dropout), dropout=self.dropout,
                                        rezero=False)
        self.encodes = clones(self.encodelayer, depth)

    def forward(self, entitys, nonZeroRows, mask: Tensor = None):
        entity_reps = torch.zeros([entitys.shape[0], entitys.shape[-1]])
        for ind, mentions in enumerate(entitys):
            mention = mentions[nonZeroRows[ind]]
            for encode in self.encodes:
                mention = encode(mention, mask)
            entity_rep = torch.logsumexp(mention, dim=0)
            entity_reps[ind, :] = entity_rep
        return entity_reps
