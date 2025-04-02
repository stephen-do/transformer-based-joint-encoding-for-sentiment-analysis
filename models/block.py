import torch.nn as nn
from models.modules.self_attention import SelfAttention
from models.modules.self_guilded_attention import SelfGuildedAttention
from models.modules.attention_flattening import AttentionFlattening
from models.layers.normalization import NormalizationLayer


class Block(nn.Module):
    def __init__(self, args, i):
        super(Block, self).__init__()
        self.args = args
        self.sa1 = SelfAttention(args)
        self.sa3 = SelfGuildedAttention(args)

        self.last = (i == args.layer-1)
        if not self.last:
            self.att_lang = AttentionFlattening(args, args.lang_seq_len, merge=False)
            self.att_audio = AttentionFlattening(args, args.audio_seq_len, merge=False)
            self.norm_l = NormalizationLayer(args.hidden_size)
            self.norm_i = NormalizationLayer(args.hidden_size)
            self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, x, x_mask, y, y_mask):

        ax = self.sa1(x, x_mask)
        ay = self.sa3(y, x, y_mask, x_mask)

        x = ax + x
        y = ay + y

        if self.last:
            return x, y

        ax = self.att_lang(x, x_mask)
        ay = self.att_audio(y, y_mask)

        return self.norm_l(x + self.dropout(ax)), \
               self.norm_i(y + self.dropout(ay))
