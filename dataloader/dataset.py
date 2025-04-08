import torch
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
import pandas as pd
import numpy as np
from argparse import Namespace


class MOSIDataset(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    #  text_context_length
    #  audio_context_length

    def __init__(self, args: Namespace, csv_path, audio_directory, mode, text_context_length=2, audio_context_length=1):
        df = pd.read_csv(csv_path)
        invalid_files = ['3aIQUQgawaI/12.wav', '94ULum9MYX0/2.wav', 'mRnEJOLkhp8/24.wav', 'aE-X_QdDaqQ/3.wav',
                         '94ULum9MYX0/11.wav', 'mRnEJOLkhp8/26.wav']
        for f in invalid_files:
            video_id = f.split('/')[0]
            clip_id = f.split('/')[1].split('.')[0]
            df = df[~((df['video_id'] == video_id) & (df['clip_id'] == int(clip_id)))]

        df = df[df['mode'] == mode].sort_values(by=['video_id', 'clip_id']).reset_index()

        # store labels
        self.targets_M = df['label']

        # store texts
        df['text'] = df['text'].str[0] + df['text'].str[1::].apply(lambda x: x.lower())
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        # store audio
        self.audio_file_paths = []
        ## loop through the csv entries
        for i in range(0, len(df)):
            file_name = str(df['video_id'][i]) + '/' + str(df['clip_id'][i]) + '.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                          do_normalize=True, return_attention_mask=True)
        self.l_max_len = args.lang_seq_len
        self.a_max_len = args.audio_seq_len
        self.a_feat_size = args.audio_feat_size


    def __getitem__(self, index):
        # load text
        text = str(self.texts[index])

        # tokenize text
        tokenized_text = self.tokenizer(
            text,
            max_length=self.l_max_len,
            padding="max_length",  # Pad to the specified max_length.
            truncation=True,  # Truncate to the specified max_length.
            add_special_tokens=True,  # Whether to insert [CLS], [SEP], <s>, etc.
            return_attention_mask=True
        )

        # load audio
        sound, _ = torchaudio.load(self.audio_file_paths[index])
        sound_data = torch.mean(sound, dim=0, keepdim=False)

        # extract audio features
        features = self.feature_extractor(sound_data, sampling_rate=16000, max_length=self.a_max_len*self.a_feat_size,
                                          truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()

        return {  # text
            "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
            # audio
            "audio_inputs": audio_features.view(self.a_max_len, self.a_feat_size),
            # labels
            "targets": torch.tensor(self.targets_M[index], dtype=torch.float),
        }

    def __len__(self):
        return len(self.targets_M)
