import glob
import itertools
import random
import torch
from .utils import read_config, dataset_split, process_mp3, init_melspectrogram, read_csv
from torch.utils.data import Dataset, DataLoader

class CreateClassifyDataset:
    def __init__(self, config):
        self.mp3_folder = config['mp3_folder']

    def __add_to_dict(self, val, mappings):
        if val not in mappings:
            mappings[val] = len(mappings)

    def create(self):
        all_mp3s = glob.glob(f"{self.mp3_folder}/*.mp3")
        filepaths =  [filepath.replace("\\", "/").split("/")[-1] for filepath in all_mp3s]

        dataset = []
        tone_dict = {}
        pinyin_dict = {}
        labels_dict = {}

        for file in filepaths:
            pinyin_tone = file.split("_")[0]
            pinyin = pinyin_tone[:-1]
            tone = pinyin_tone[-1]

            self.__add_to_dict(pinyin_tone, labels_dict)
            self.__add_to_dict(pinyin, pinyin_dict)
            self.__add_to_dict(tone, tone_dict)

            dataset.append([file, pinyin, tone, pinyin_tone])

        mappings_dict = {"pinyin_mappings": pinyin_dict, "tone_mappings": tone_dict, "label_mappings": labels_dict}
        return dataset, mappings_dict


class ClassifyDataset(Dataset):
    def __init__(self, dataset: list, config, mappings: dict):
        self.mp3_folder = config['mp3_folder']
        self.sr = config["sampling_rate"]
        self.mappings_dict = mappings
        self.dataset = dataset
        self.transform = init_melspectrogram(self.sr, 2048)
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __encode(self, val, mapping):
        one_hot_arr = [0] * len(mapping)
        one_hot_arr[mapping[val]] = 1
        return one_hot_arr

    def __getitem__(self, idx: int):
        data = self.dataset[idx]

        if data[0] not in self.cache:
            self.cache[data[0]] = process_mp3(f"{self.mp3_folder}/{data[0]}") #, self.sr, self.transform)

        mfcc = self.cache[data[0]]
        pinyin_tone_one_hot = self.__encode(data[3], self.mappings_dict["label_mappings"])
        pinyin_one_hot = self.__encode(data[1], self.mappings_dict["pinyin_mappings"])
        tone_one_hot = self.__encode(str(data[2]), self.mappings_dict["tone_mappings"])

        return mfcc, torch.Tensor(pinyin_one_hot), torch.Tensor(tone_one_hot), torch.Tensor(pinyin_tone_one_hot)

class CreateClassifyDataloader:
    def __init__(self, config: str):
        self.config = config
        self.val_split = config["val_split"]
        self.batch_size = config["batch_size"]
        self.train_split = config["train_split"]
        self.full_dataset, self.mappings = CreateClassifyDataset(config).create()

    def create(self):
        #train = read_csv(self.config["class_train_csv"])
        #val   = read_csv(self.config["class_val_csv"])
        #test  = read_csv(self.config["class_test_csv"])

        train, val, test = dataset_split(self.full_dataset, self.train_split, self.val_split)
        train = ClassifyDataset(train, self.config, self.mappings)
        test = ClassifyDataset(test, self.config, self.mappings)
        val = ClassifyDataset(val, self.config, self.mappings)

        train = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        test = DataLoader(test, batch_size=self.batch_size, shuffle=True)
        val = DataLoader(val, batch_size=self.batch_size, shuffle=True)

        return train, val, test, self.mappings

if __name__ == "__main__":
    json_path = "../config.json"
    json = read_config(json_path)
    train, val, test, mappings = CreateClassifyDataloader(json).create()
    next(iter(train))
