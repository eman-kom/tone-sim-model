import torch
import json
from Model import SiameseModel
from create_datasets.CreateClassificationDataset import CreateClassifyDataset
import argparse
import librosa
import numpy as np
import torch
from torch import nn
import math

def process_mp3(audio_fname, n_mfcc=128, max_pad=60):
    audio, sample_rate = librosa.core.load(audio_fname)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    pad_width = max_pad - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    mfcc = torch.tensor(mfcc).unsqueeze(0)
    mfcc = mfcc.unsqueeze(0)
    return mfcc

def find_dist(ref_embeds, embeds):
    embeds = embeds.squeeze()
    ref_embeds = ref_embeds.squeeze()
    pdist = nn.PairwiseDistance(p=2)
    dist = 1 - pdist(embeds, ref_embeds).item()
    return dist

def find_pinyin_tone(container, out, type):
    _, max_idxs = torch.max(out.data, 1)
    decoder = self.decoder[type]
    max_idxs = max_idxs.detach().cpu().numpy().tolist()
    container.extend([decoder[idx] for idx in max_idxs])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config.json")
    parser.add_argument("-f1", "--file1")
    parser.add_argument("-f2", "--file2")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    model = SiameseModel(config["transfer_model"], 4, 410, config["device"])
    weights = torch.load(config["inference_weights"], map_location=config["device"], weights_only=True)
    model.load_state_dict(weights)

    f1 = process_mp3(args.file1)
    f2 = process_mp3(args.file2)

    model.eval()
    with torch.no_grad():
        changed_embeddings, correct_embeddings, pinyin_tone = model(f1, f2)

    dist = find_dist(correct_embeddings, changed_embeddings)

    pinyin_preds, tone_preds = pinyin_tone

    tone_preds = tone_preds.squeeze()
    _, tone = torch.max(tone_preds, 0)
    tone = tone.item() + 1

    _, mappings = CreateClassifyDataset(config).create()
    mappings = dict(zip(mappings["pinyin_mappings"].values(), mappings["pinyin_mappings"].keys()))
    pinyin_preds = pinyin_preds.squeeze()
    _, pinyin_idx = torch.max(pinyin_preds, 0)
    pinyin_idx = pinyin_idx.item()

    print("---")
    print(f"Pinyin Prediction: {mappings[pinyin_idx]}")
    print(f"Tone Prediction  : {tone}")
    print(f"Similarity Score : {round(dist, 4)}")
