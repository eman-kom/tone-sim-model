import glob
import csv
import json
from utils import dataset_split, save_csv

class CreateClassifyDataset:
    def __init__(self, config):
        self.mp3_folder = config['mp3_folder']
        self.csv_folder = config['csv_folder']
        self.val_split = config['val_split']
        self.train_split = config['train_split']


    def __add_to_dict(self, val, mappings):
        if val not in mappings:
            mappings[val] = len(mappings)


    def create(self):
        all_mp3s = glob.glob(f"{self.mp3_folder}/*.mp3")
        filepaths =  [filepath.replace("\\", "/").split("/")[-1] for filepath in all_mp3s]
        filepaths.sort()

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


    def __save_dict(self, filename: str, mappings: dict):
        with open(f"{self.csv_folder}/{filename}.json", "w") as f:
            json.dump(mappings, f, indent=4)

        print(f"[*] File Created: {self.csv_folder}/{filename}.json")


    def save(self):
        dataset, mappings_dict = self.create()

        self.__save_dict("tones", mappings_dict["tone_mappings"])
        self.__save_dict("pinyins", mappings_dict["pinyin_mappings"])
        self.__save_dict("pinyin_tone", mappings_dict["label_mappings"])

        train, val, test = dataset_split(dataset, self.train_split, self.val_split)
        fields = ["file", "pinyin", "tone", "pinyin_tone"]

        save_csv(self.csv_folder, "cl_train", fields, train)
        save_csv(self.csv_folder, "cl_val", fields, val)
        save_csv(self.csv_folder, "cl_test", fields, test)
