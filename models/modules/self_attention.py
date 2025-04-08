import torch.nn as nn
from models.layers.normalization import NormalizationLayer
from models.modules.multi_head_attention import MultiHeadAttention
from models.layers.feed_forward_net import FeedForwardNet
from argparse import Namespace


class SelfAttention(nn.Module):
    def __init__(self, args: Namespace):
        super(SelfAttention, self).__init__()

        self.mhatt = MultiHeadAttention(args)
        self.ffn = FeedForwardNet(args)

        self.dropout1 = nn.Dropout(args.dropout_rate)
        self.norm1 = NormalizationLayer(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_rate)
        self.norm2 = NormalizationLayer(args.hidden_size)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y
