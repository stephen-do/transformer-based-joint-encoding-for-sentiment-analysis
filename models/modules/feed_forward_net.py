import torch.nn as nn
from models.layers.multi_layer_perceptron import MultiLayerPerceptron
from models.layers.normalization import NormalizationLayer


class FeedForwardNet(nn.Module):
    def __init__(self, args):
        super(FeedForwardNet, self).__init__()

        self.mlp = MultiLayerPerceptron(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=args.hidden_size,
            dropout_r=args.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)
