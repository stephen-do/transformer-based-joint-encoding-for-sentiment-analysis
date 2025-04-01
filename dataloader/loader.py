from torch.utils.data import DataLoader
from dataloader.dataset import MOSIDataset


def loader(batch_size, text_context_length=2, audio_context_length=1):
    csv_path = 'dataset/CMU-MOSI/label.csv'
    audio_file_path = "dataset/CMU-MOSI/wav"
    train_data = MOSIDataset(csv_path, audio_file_path, 'train', text_context_length=text_context_length,
                             audio_context_length=audio_context_length)
    test_data = MOSIDataset(csv_path, audio_file_path, 'test', text_context_length=text_context_length,
                            audio_context_length=audio_context_length)
    val_data = MOSIDataset(csv_path, audio_file_path, 'valid', text_context_length=text_context_length,
                           audio_context_length=audio_context_length)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader
