import glob
import torch
import random
import itertools
from torch.utils.data import Dataset, DataLoader
from .utils import read_config, dataset_split, process_mp3, init_melspectrogram, read_csv

class CreateChangeDataset:
    def __init__(self, config):
        self.mp3_folder = config['mp3_folder']

    def __extract_pinyin_tone(self, filename: str):
        pinyin_tone, _, _ = filename.split("_")
        return pinyin_tone

    def __combine_same_tones(self, filepaths: list, curr_tone: str) -> list:
        same_tones = []
        combinations = []
        pair_tones = lambda x: [[ground_truth, changed, 1] for ground_truth, changed in itertools.combinations(x, 2)]
        
        for file in filepaths:
            pinyin_tone = self.__extract_pinyin_tone(file)
        
            if pinyin_tone == curr_tone:
                same_tones.append(file)
            else:
                combinations.extend(pair_tones(same_tones))
                curr_tone = pinyin_tone
                same_tones = [file]

        combinations.extend(pair_tones(same_tones))
        return combinations
    
    def __generate_mismatches(self, combinations: list) -> list:
        mismatch_count = len(combinations)
        mismatches = []

        for i in range(mismatch_count):
            ground_truth = combinations[i][0]
            pinyin_truth = self.__extract_pinyin_tone(ground_truth)

            while True:
                changed = random.choice(combinations)[0]
                pinyin_changed = self.__extract_pinyin_tone(changed)

                if pinyin_truth != pinyin_changed:
                    mismatches.append([ground_truth, changed, 0])
                    break

        return mismatches

    def create(self, init_tone="a1"):
        all_mp3s = glob.glob(f"{self.mp3_folder}/*.mp3")
        filepaths =  [filepath.replace("\\", "/").split("/")[-1] for filepath in all_mp3s]
        filepaths.sort()
        all_combinations = self.__combine_same_tones(filepaths, init_tone)
        mismatches = self.__generate_mismatches(all_combinations)
        all_combinations.extend(mismatches)

        return all_combinations


class ChangeDataset(Dataset):
    def __init__(self, dataset: list, configs):
        config = configs
        self.mp3_folder = config['mp3_folder']
        self.sr = config["sampling_rate"]
        self.dataset = dataset
        self.cache = {}
        self.transform = init_melspectrogram(self.sr, 2048)

    def __len__(self):
        return len(self.dataset)

    def __process_mp3(self, file):
        if file not in self.cache:
            self.cache[file] = process_mp3(file)#, self.sr, self.transform)

        return self.cache[file]

    def __getitem__(self, idx: int):
        data = self.dataset[idx]
        gt = self.__process_mp3(f"{self.mp3_folder}/{data[0]}")
        ch = self.__process_mp3(f"{self.mp3_folder}/{data[1]}")
        return gt, ch, torch.Tensor([data[2]]).squeeze()

class CreateChangeDataloader:
    def __init__(self, config):
        self.config = config
        self.val_split = config["val_split"]
        self.batch_size = config["batch_size"]
        self.train_split = config["train_split"]
        self.full_dataset = CreateChangeDataset(config).create()

    def create(self):
        #train = read_csv(self.config["cd_train_csv"])
        #val   = read_csv(self.config["cd_val_csv"])
        #test  = read_csv(self.config["cd_test_csv"])

        random.shuffle(self.full_dataset)
        train, val, test = dataset_split(self.full_dataset, self.train_split, self.val_split)
        train = ChangeDataset(train, self.config)
        test = ChangeDataset(test, self.config)
        val = ChangeDataset(val, self.config)

        train = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        test = DataLoader(test, batch_size=self.batch_size, shuffle=True)
        val = DataLoader(val, batch_size=self.batch_size, shuffle=True)

        return train, val, test

if __name__ == "__main__":
    json_path = "../desktop.json"
    json_path = read_config(json_path)
    train, val, test = CreateChangeDataloader(json_path).create()
    a = next(iter(train))
    print(a[0].shape)
