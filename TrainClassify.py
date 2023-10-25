import argparse
import torch
import json
from tqdm import tqdm
from torch import nn, optim
from Model import ClassificationModel
from torch.utils.tensorboard import SummaryWriter
from create_datasets.CreateClassificationDataset import CreateClassifyDataloader

class ClassifyTrain:
    def __init__(self, config):
        self.train, self.val, self.test, self.mappings = CreateClassifyDataloader(config).create()
        n_tones = len(self.mappings["tone_mappings"])
        n_pinyins = len(self.mappings["pinyin_mappings"])

        self.device = config["device"]
        self.save_models_dir = config['saved_models_folder']

        self.decoder = {}
        swap_key_values = lambda mappings: dict(zip(mappings.values(), mappings.keys()))
        self.decoder["tone_mappings_decode"] = swap_key_values(self.mappings["tone_mappings"])
        self.decoder["pinyin_mappings_decode"] = swap_key_values(self.mappings["pinyin_mappings"])

        self.epochs = config["epochs"]
        self.tone_criterion = nn.CrossEntropyLoss()
        self.pinyin_criterion = nn.CrossEntropyLoss()
        self.model = ClassificationModel(n_tones, n_pinyins).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.9, verbose=False,)

    def __criterion(self, tone_out, tone_tgt, pinyin_out, pinyin_tgt):
        tone_loss = self.tone_criterion(tone_out, tone_tgt)
        pinyin_loss = self.pinyin_criterion(pinyin_out, pinyin_tgt)
        return tone_loss + pinyin_loss

    def __decode_one_hot(self, container, out, type):
        _, max_idxs = torch.max(out.data, 1)
        decoder = self.decoder[type]
        max_idxs = max_idxs.detach().cpu().numpy().tolist()
        container.extend([decoder[idx] for idx in max_idxs])

    def __accuracy(self, preds, truth):
        correct_matches = 0
        for a, b in zip(preds, truth):
            correct_matches += a == b
        return correct_matches / len(truth)

    def __runner(self, dataset, isTrain=True):
        epoch_loss = 0
        y_true_tone, y_pred_tone = [], []
        y_true_pinyin, y_pred_pinyin = [], []

        for file, pinyin_tgt, tone_tgt, _ in tqdm(dataset):
            file, pinyin_tgt, tone_tgt = file.to(self.device), pinyin_tgt.to(self.device), tone_tgt.to(self.device)
            pinyin_out, tone_out = self.model(file)
            loss = self.__criterion(tone_out, tone_tgt, pinyin_out, pinyin_tgt)

            if isTrain:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            self.__decode_one_hot(y_pred_tone, tone_out, "tone_mappings_decode")
            self.__decode_one_hot(y_true_tone, tone_tgt, "tone_mappings_decode")
            self.__decode_one_hot(y_pred_pinyin, pinyin_out, "pinyin_mappings_decode")
            self.__decode_one_hot(y_true_pinyin, pinyin_tgt, "pinyin_mappings_decode")

        epoch_loss = epoch_loss / len(dataset)
        tone_acc = self.__accuracy(y_pred_tone, y_true_tone)
        pinyin_acc = self.__accuracy(y_pred_pinyin, y_true_pinyin)

        return epoch_loss, tone_acc, pinyin_acc

    def run(self):
        min_val_loss = float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            train_loss, train_tone, train_pinyin = self.__runner(self.train)

            self.model.eval()
            with torch.no_grad():
                val_loss, val_tone, val_pinyin = self.__runner(self.val, isTrain=False)

            print(f"[+] Epoch: {epoch}")
            print(f"Train | Loss: {round(train_loss, 4)}, Tone Acc: {round(train_tone, 4)}, Pinyin Acc: {round(train_pinyin, 4)}")
            print(f"Val   | Loss: {round(val_loss, 4)}, Tone Acc: {round(val_tone, 4)}, Pinyin Acc: {round(val_pinyin, 4)}")

            self.scheduler.step()
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Tone_Acc/train", train_tone, epoch)
            writer.add_scalar("Pinyin_Acc/train", train_pinyin, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Tone_Acc/validation", val_tone, epoch)
            writer.add_scalar("Pinyin_Acc/validation", val_pinyin, epoch)

            if val_loss < min_val_loss:
                torch.save(self.model.state_dict(), f'{self.save_models_dir}/best_multitask_model.pth')
                val_loss = min_val_loss

        self.model.eval()
        with torch.no_grad():
            _, test_tone, test_pinyin = self.__runner(self.test, isTrain=False)
            writer.add_text('Test Tone Acc', str(test_tone))
            writer.add_text('Test Pinyin Acc', str(test_pinyin))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", default="./config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    writer = SummaryWriter(log_dir=config['log_dir'])
    ClassifyTrain(config).run()

    writer.flush()
    writer.close()
