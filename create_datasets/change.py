import glob
import itertools
import random
import csv
from utils import dataset_split, save_csv

class CreateChangeDataset:
    def __init__(self, config):
        self.mp3_folder = config['mp3_folder']
        self.csv_folder = config['csv_folder']
        self.val_split = config['val_split']
        self.train_split = config['train_split']

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
                    mismatches.append([ground_truth, changed, -1])
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

    def save(self):
        dataset = self.create()
        train, val, test = dataset_split(dataset, self.train_split, self.val_split)
        fields = ["reference", "user_input", "similarity"]

        save_csv(self.csv_folder, "ch_train", fields, train);
        save_csv(self.csv_folder, "ch_val", fields, val);
        save_csv(self.csv_folder, "ch_test", fields, test);
