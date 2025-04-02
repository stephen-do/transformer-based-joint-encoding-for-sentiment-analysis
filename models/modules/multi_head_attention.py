import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)