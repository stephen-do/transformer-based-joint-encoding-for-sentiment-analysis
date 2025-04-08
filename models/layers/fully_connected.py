import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0., use_relu=True):
        super(FullyConnectedLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x
