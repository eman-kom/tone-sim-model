import json
import torch
import torchaudio

def read_config(fp: str):
    with open(fp, "r") as f:
        config = json.load(f)

    return config

def dataset_split(dataset: list, train_split: float, val_split: float):
    train_split = int(train_split * len(dataset))
    train = dataset[:train_split]

    val_split = int(val_split * len(dataset))
    val = dataset[train_split:train_split+val_split]

    test = dataset[train_split+val_split:]

    return train, val, test

def init_melspectrogram(sampling_rate: int, n_fft: int, win_length=None, hop_length=None):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length
    )
    return transform

import librosa
import numpy as np

def process_mp3(audio_fname, n_mfcc=128, max_pad=60):
    audio, sample_rate = librosa.core.load(audio_fname)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    pad_width = max_pad - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    mfcc = torch.tensor(mfcc).unsqueeze(0)
    return mfcc

import pandas as pd
def read_csv(fp):
    return pd.read_csv(fp).values.tolist()


def process_mp31(file, sampling_rate: int, transform):
    y, sr = torchaudio.load(file)
    y = torch.mean(y, 0)
    y = torchaudio.functional.resample(y, sr, sampling_rate)

    # avg length of all mp3s
    AVG_LEN = 31280
    if y.shape[0] < AVG_LEN:
        y = torch.nn.functional.pad(y, (0, AVG_LEN-y.shape[0]), "constant", 0)
    else:
        y = y[:AVG_LEN]

    y = transform(y)
    y = torch.unsqueeze(y, axis=0)
    return y
