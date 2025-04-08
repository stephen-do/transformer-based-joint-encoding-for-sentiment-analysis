import torch
import os
import argparse
import random
from dataloader.loader import loader
from models.model import Model
from train import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_encode_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=96)
    parser.add_argument('--mid_size', type=int, default=2048)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--multi_head', type=int, default=2)
    parser.add_argument('--word_embed_size', type=int, default=300)
    parser.add_argument('--ans_size', type=int, default=1)

    # Data
    parser.add_argument('--lang_seq_len', type=int, default=60)
    parser.add_argument('--audio_seq_len', type=int, default=60)
    parser.add_argument('--audio_feat_size', type=int, default=80)

    # Training
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=99)
    parser.add_argument('--opt_params', type=str, default="{'betas': '(0.9, 0.98)', 'eps': '1e-9'}")
    parser.add_argument('--lr_base', type=float, default=0.00005)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--lr_decay_times', type=int, default=2)
    parser.add_argument('--grad_norm_clip', type=float, default=-1)
    parser.add_argument('--eval_start', type=int, default=1)
    parser.add_argument('--early_stop', type=int, default=3)
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999999))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, val_loader = loader(args, args.batch_size)

    vocab_size = train_loader.dataset.tokenizer.vocab_size
    pretrained_emb = torch.randn(vocab_size, args.word_embed_size)

    model = Model(args, vocab_size, pretrained_emb)
    if not os.path.exists(os.path.join(args.output, args.name)):
        os.makedirs(os.path.join(args.output, args.name))
    eval_accuracies = train(model, train_loader, val_loader, args)


if __name__ == "__main__":
    main()
