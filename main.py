import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from dataloader.loader import loader
from models.model import Model


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            text_tokens = batch["text_tokens"].to(device)
            text_masks = batch["text_masks"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(text_tokens, audio_inputs, text_masks)
            loss = criterion(outputs.squeeze(), targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            text_tokens = batch["text_tokens"].to(device)
            text_masks = batch["text_masks"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(text_tokens, audio_inputs, text_masks)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--word_embed_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=960000)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--ans_size", type=int, default=1)
    parser.add_argument('--multi_head', type=int, default=8)
    parser.add_argument('--ff_size', type=int, default=2048)
    parser.add_argument('--dropout_r', type=float, default=0.1)
    parser.add_argument('--lang_seq_len', type=int, default=60)
    parser.add_argument('--audio_seq_len', type=int, default=60)
    parser.add_argument('--audio_feat_size', type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, val_loader = loader(args.batch_size)

    vocab_size = train_loader.dataset.tokenizer.vocab_size
    pretrained_emb = torch.randn(vocab_size, args.word_embed_size)  # Thay bằng embedding thực tế

    model = Model(args, vocab_size, pretrained_emb)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)


if __name__ == "__main__":
    main()
