import torch
from utils import load_csv, read_json, process_mp3
from torch.utils.data import Dataset, DataLoader

class ClassifyDataset(Dataset):
    def __init__(self, config: dict, filename: str):
        self.dataset = load_csv(config["csv_folder"], filename)
        self.mp3_folder = config["mp3_folder"]
        self.sr = config["sampling_rate"]
        self.cache = {}

        self.tones = read_json(f"{config['csv_folder']}/tones.json")
        self.pinyins = read_json(f"{config['csv_folder']}/pinyins.json")
        self.pinyin_tone = read_json(f"{config['csv_folder']}/pinyin_tone.json")


    def __len__(self):
        return len(self.dataset)


    def __encode(self, val, mapping):
        one_hot_arr = [0] * len(mapping)
        one_hot_arr[mapping[val]] = 1
        return one_hot_arr


    def __getitem__(self, idx: int):
        data = self.dataset[idx]
        
        if data[0] not in self.cache:
            filepath = f"{self.mp3_folder}/{data[0]}"
            self.cache[data[0]] = process_mp3(filepath)
        
        mfcc = self.cache[data[0]]
        pinyin_one_hot = self.__encode(data[1], self.pinyins)
        tone_one_hot = self.__encode(str(data[2]), self.tones)
        pinyin_tone_one_hot = self.__encode(data[3], self.pinyin_tone)

        return mfcc, torch.Tensor(pinyin_one_hot), torch.Tensor(tone_one_hot), torch.Tensor(pinyin_tone_one_hot)


class ClassifyDataloader():
    def __init__(self, config: dict):
        self.train = ClassifyDataset(config, "cl_train")
        self.val = ClassifyDataset(config, "cl_val")
        self.test = ClassifyDataset(config, "cl_test")
        self.batch_size = config["batch_size"]

        self.tones = read_json(f"{config['csv_folder']}/tones.json")
        self.pinyins = read_json(f"{config['csv_folder']}/pinyins.json")


    def create(self):
        train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test, batch_size=self.batch_size, shuffle=True)

        swap_key_values = lambda mappings: dict(zip(mappings.values(), mappings.keys()))
        pinyin_mappings = swap_key_values(self.pinyins)
        tone_mappings = swap_key_values(self.tones)

        return train_loader, val_loader, test_loader, pinyin_mappings, tone_mappings
