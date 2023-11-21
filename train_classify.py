from create_dataloader.classify import ClassifyDataloader
from Model import ClassificationModel
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import read_json, calculate_accuracy
import argparse
import torch

class TrainClassify:
    def __init__(self, config: dict):
        self.train, self.val, self.test, self.pinyin_mappings, \
        self.tone_mappings = ClassifyDataloader(config).create()

        n_pinyins = len(self.pinyin_mappings)
        n_tones = len(self.tone_mappings)

        self.epochs = config["epochs"]
        self.device = config["device"]
        self.name = config["pretrained_model_name"]
        self.models_folder = config['models_folder']

        self.tone_criterion = nn.CrossEntropyLoss()
        self.pinyin_criterion = nn.CrossEntropyLoss()
        self.model = ClassificationModel(n_tones, n_pinyins).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)


    def __criterion(self, tone_out: torch.Tensor, tone_tgt: torch.Tensor, 
                    pinyin_out: torch.Tensor, pinyin_tgt: torch.Tensor) -> torch.float32:
        """
        Finds the loss
        """
        tone_loss = self.tone_criterion(tone_out, tone_tgt)
        pinyin_loss = self.pinyin_criterion(pinyin_out, pinyin_tgt)
        return tone_loss + pinyin_loss


    def __decode_one_hot(self, container: list, output: torch.Tensor, mappings_dict: dict) -> None:
        """
        Finds the most likely mapping from prediction array
        """
        _, max_idxs = torch.max(output.data, 1)
        max_idxs = max_idxs.detach().cpu().numpy().tolist()
        container.extend([mappings_dict[idx] for idx in max_idxs])


    def __runner(self, dataset, isTrain=True) -> tuple:
        """
        Passes the dataset through the model
        """
        epoch_loss = 0
        y_true_tone, y_pred_tone = [], []
        y_true_pinyin, y_pred_pinyin = [], []

        for mfcc, pinyin_tgt, tone_tgt, _ in tqdm(dataset):
            mfcc = mfcc.to(self.device)
            tone_tgt = tone_tgt.to(self.device)
            pinyin_tgt = pinyin_tgt.to(self.device)

            pinyin_out, tone_out = self.model(mfcc)
            loss = self.__criterion(tone_out, tone_tgt, pinyin_out, pinyin_tgt)

            if isTrain:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            self.__decode_one_hot(y_pred_tone, tone_out, self.tone_mappings)
            self.__decode_one_hot(y_true_tone, tone_tgt, self.tone_mappings)
            self.__decode_one_hot(y_pred_pinyin, pinyin_out, self.pinyin_mappings)
            self.__decode_one_hot(y_true_pinyin, pinyin_tgt, self.pinyin_mappings)

        epoch_loss = epoch_loss / len(dataset)
        tone_acc = calculate_accuracy(y_pred_tone, y_true_tone)
        pinyin_acc = calculate_accuracy(y_pred_pinyin, y_true_pinyin)

        return epoch_loss, tone_acc, pinyin_acc


    def run(self) -> None:
        """
        Entrypoint to train the model
        """
        min_val_loss = float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            train_loss, train_tone, train_pinyin = self.__runner(self.train)

            self.model.eval()
            with torch.no_grad():
                val_loss, val_tone, val_pinyin = self.__runner(self.val, isTrain=False)

            print(f"[+] Epoch: {epoch+1}")
            print(f"Train | Loss: {round(train_loss, 4)}, Tone Acc: {round(train_tone, 4)}, Pinyin Acc: {round(train_pinyin, 4)}")
            print(f"Val   | Loss: {round(val_loss, 4)}, Tone Acc: {round(val_tone, 4)}, Pinyin Acc: {round(val_pinyin, 4)}")

            self.scheduler.step()
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Tone_Acc/train", train_tone, epoch)
            writer.add_scalar("Pinyin_Acc/train", train_pinyin, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Tone_Acc/validation", val_tone, epoch)
            writer.add_scalar("Pinyin_Acc/validation", val_pinyin, epoch)

            # Uses the minimum val loss as best model criterion
            if val_loss < min_val_loss:
                torch.save(self.model.state_dict(), f'{self.models_folder}/{self.name}')
                min_val_loss = val_loss

        self.model.eval()
        with torch.no_grad():
            test_loss, test_tone, test_pinyin = self.__runner(self.test, isTrain=False)
        
        print("[+] Testing Phase")
        print(f"Loss: {test_loss}")
        print(f"Tone Acc: {test_tone}")
        print(f"Test Pinyin Acc: {test_pinyin}")

        writer.add_text('Test Loss', str(test_loss))
        writer.add_text('Test Tone Acc', str(test_tone))
        writer.add_text('Test Pinyin Acc', str(test_pinyin))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.json")
    args = parser.parse_args()

    writer = SummaryWriter()
    config = read_json(args.config)

    TrainClassify(config).run()

    writer.flush()
    writer.close()
