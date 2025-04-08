import torch.nn as nn
from models.layers.fully_connected import FullyConnectedLayer


class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_rate=0., use_relu=True):
        super(MultiLayerPerceptron, self).__init__()

        self.fc = FullyConnectedLayer(in_size, mid_size, dropout_rate=dropout_rate, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))
