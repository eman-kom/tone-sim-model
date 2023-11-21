import torch
from utils import load_csv, process_mp3
from torch.utils.data import Dataset, DataLoader

class ChangeDataset(Dataset):
    def __init__(self, config: dict, filename: str):
        self.dataset = load_csv(config["csv_folder"], filename)
        self.mp3_folder = config["mp3_folder"]
        self.cache = {}

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset
        """
        return len(self.dataset)

    def __process_mp3(self, file: str) -> torch.Tensor:
        """
        Caches and process the mp3 file
        """
        if file not in self.cache:
            self.cache[file] = process_mp3(file)

        return self.cache[file]

    def __getitem__(self, idx: int) -> tuple:
        data = self.dataset[idx]
        reference = self.__process_mp3(f"{self.mp3_folder}/{data[0]}")
        user_input = self.__process_mp3(f"{self.mp3_folder}/{data[1]}")
        similarity = int(data[2])
        return reference, user_input, torch.Tensor([similarity]).squeeze()


class ChangeDataloader():
    def __init__(self, config: dict):
        self.train = ChangeDataset(config, "ch_train")
        self.val = ChangeDataset(config, "ch_val")
        self.test = ChangeDataset(config, "ch_test")
        self.batch_size = config["batch_size"]


    def create(self) -> tuple:
        """
        Initialises train, test and validation dataloaders
        """
        train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test, batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader, test_loader
