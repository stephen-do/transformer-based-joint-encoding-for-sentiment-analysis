import torch.nn as nn
from models.layers.normalization import NormalizationLayer
from models.modules.multi_head_attention import MultiHeadAttention
from models.modules.feed_forward_net import FeedForwardNet
from argparse import Namespace


class SelfGuildedAttention(nn.Module):
    def __init__(self, args: Namespace):
        super(SelfGuildedAttention, self).__init__()

        self.mhatt1 = MultiHeadAttention(args)
        self.mhatt2 = MultiHeadAttention(args)
        self.ffn = FeedForwardNet(args)

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = NormalizationLayer(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = NormalizationLayer(args.hidden_size)

        self.dropout3 = nn.Dropout(args.dropout_r)
        self.norm3 = NormalizationLayer(args.hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x
