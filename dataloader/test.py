import torch
from dataset import MOSIDataset
from torch.utils.data import DataLoader
import argparse
parser = argparse.ArgumentParser()
# Data
parser.add_argument('--lang_seq_len', type=int, default=60)
parser.add_argument('--audio_seq_len', type=int, default=60)
parser.add_argument('--audio_feat_size', type=int, default=80)
args = parser.parse_args()
csv_path = "dataset/CMU-MOSI/label.csv"
audio_file_path = "dataset/CMU-MOSI/wav"
text_context_length = 2
audio_context_length = 1
batch_size = 4

train_dataset = MOSIDataset(args, csv_path, audio_file_path, 'train', text_context_length, audio_context_length)

print(f"Tổng số mẫu trong tập train: {len(train_dataset)}")
sample = train_dataset[0]
print("Dữ liệu mẫu:")
for key, value in sample.items():
    print(f"{key}: {value if isinstance(value, torch.Tensor) else value}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for batch in train_loader:
    print("\nBatch dữ liệu:")
    for key, value in batch.items():
        print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else value}")
    break
