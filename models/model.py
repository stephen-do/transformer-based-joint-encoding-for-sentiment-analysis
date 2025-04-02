import torch
import torch.nn as nn
from models.layers.normalization import NormalizationLayer
from models.modules.attention_flattening import AttentionFlattening
from models.block import Block

def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


class Model(nn.Module):
    def __init__(self, args, vocab_size, pretrained_emb):
        super(Model, self).__init__()

        self.args = args

        # LSTM
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=args.word_embed_size
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm_x = nn.LSTM(
            input_size=args.word_embed_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # self.lstm_y = nn.LSTM(
        #     input_size=args.audio_feat_size,
        #     hidden_size=args.hidden_size,
        #     num_layers=1,
        #     batch_first=True
        # )

        # Feature size to hid size
        self.adapter = nn.Linear(args.audio_feat_size, args.hidden_size)

        # Encoder blocks
        self.enc_list = nn.ModuleList([Block(args, i) for i in range(args.layer)])

        # Flattenting features before proj
        self.attflat_img  = AttentionFlattening(args, 1, merge=True)
        self.attflat_lang = AttentionFlattening(args, 1, merge=True)

        # Classification layers
        self.proj_norm = NormalizationLayer(2 * args.hidden_size)
        self.proj = self.proj = nn.Linear(2 * args.hidden_size, args.ans_size)

    def forward(self, x, y, _):
        x_mask = make_mask(x.unsqueeze(2))
        y_mask = make_mask(y)

        embedding = self.embedding(x)

        x, _ = self.lstm_x(embedding)
        # y, _ = self.lstm_y(y)

        y = self.adapter(y)

        for i, dec in enumerate(self.enc_list):
            x_m, x_y = None, None
            if i == 0:
                x_m, x_y = x_mask, y_mask
            x, y = dec(x, x_m, y, x_y)

        x = self.attflat_lang(
            x,
            None
        )

        y = self.attflat_img(
            y,
            None
        )

        # Classification layers
        proj_feat = x + y
        proj_feat = self.proj_norm(proj_feat)
        ans = self.proj(proj_feat)

        return ans