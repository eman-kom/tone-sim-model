import csv
import json
import random
import librosa
import numpy as np
import torch

def read_json(filepath: str) -> dict:
    """
    Reads and loads the json file
    """
    with open(filepath, "r") as f:
        return json.load(f)

def dataset_split(dataset: list, train_split: float, val_split: float) -> tuple:
    """
    Splits the dataset into train, test and validation
    """
    random.shuffle(dataset)
    
    train_split = int(train_split * len(dataset))
    train = dataset[:train_split]

    val_split = int(val_split * len(dataset))
    val = dataset[train_split:train_split+val_split]

    test = dataset[train_split+val_split:]

    return train, val, test

def save_csv(folder: str, filename: str, fields: list, rows: list) -> None:
    """
    Saves dataset as csv
    """
    filepath = f"{folder}/{filename}.csv"

    with open(filepath, "w", newline='', encoding='utf-8') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)

    print(f"[*] File Created: {filepath}")

def load_csv(folder: str, filename: str) -> list:
    """
    Loads csv file
    """
    filepath = f"{folder}/{filename}.csv"
    with open(filepath, 'r') as f: 
        return list(csv.reader(f))[1:]

def process_mp3(audio_fname, n_mfcc=128, max_pad=60) -> torch.Tensor:
    """
    Converts the mp3 file into mfcc
    """
    audio, sample_rate = librosa.core.load(audio_fname)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    pad_width = max_pad - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    mfcc = torch.tensor(mfcc).unsqueeze(0)
    return mfcc

def calculate_accuracy(list1: list, list2: list, accuracy=0) -> float:
    """
    Finds number of true positives between lists
    """
    for a, b in zip(list1, list2):
        accuracy += a == b

    return accuracy / len(list1)

def dist_filename(filename: str, isEuclid: bool) -> str:
    """
    Returns the appropriate file prefix
    """
    if isEuclid:
        return "euclid-" + filename
    else:
        return "cosine-" + filename
