from create_dataloader.change import ChangeDataloader
from Model import SiameseModel
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import read_json, calculate_accuracy
import argparse
import json
import torch
import numpy as np

class TrainSiamese:
    def __init__(self, config):
        self.train, self.val, self.test = ChangeDataloader(config).create()

        self.device = config["device"]
        self.epochs = config["epochs"]
        self.name = config["siamese_model_name"]
        self.models_folder = config['models_folder']

        n_tones = len(read_json(f"{config['csv_folder']}/tones.json"))
        n_pinyins = len(read_json(f"{config['csv_folder']}/pinyins.json"))
        pretrained_model = f"{config['models_folder']}/{config['pretrained_model_name']}"

        self.model = SiameseModel(pretrained_model, n_tones, n_pinyins, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

        self.dist = nn.CosineSimilarity()
        self.loss_fn = nn.CosineEmbeddingLoss()


    def __contrastive_loss(self, embeddings, ref_embeddings, similarity, margin=1.0):
        return self.loss_fn(embeddings, ref_embeddings, similarity)


    def __calc_dist(self, embeddings, ref_embeddings):
        return self.dist(embeddings, ref_embeddings)


    def __runner(self, dataset, isTrain=True):
        epoch_loss = 0
        y_true_similarity, y_pred_similarity = [], []

        for reference, user_input, similarity in tqdm(dataset):
            reference = reference.to(self.device)
            user_input = user_input.to(self.device)
            similarity = similarity.to(self.device)

            pred_embeds, ref_embeds, _ = self.model(user_input, reference)
            loss = self.__contrastive_loss(pred_embeds, ref_embeds, similarity)
            
            if isTrain:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            y_true_similarity.extend(similarity)
            preds_sims = self.__calc_dist(pred_embeds, ref_embeds).detach().cpu().numpy()
            preds_sims[preds_sims > 0.5] = 1
            preds_sims[preds_sims < 0.5] = -1
            y_pred_similarity.extend(preds_sims.tolist())

        epoch_loss = epoch_loss / len(dataset)
        accuracy = calculate_accuracy(y_true_similarity, y_pred_similarity).item()

        return epoch_loss, accuracy


    def run(self):
        min_val_loss = float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            train_loss, train_acc = self.__runner(self.train)

            self.model.eval()
            with torch.no_grad():
                val_loss, val_acc = self.__runner(self.val, isTrain=False)

            print(f"[+] Epoch: {epoch}")
            print(f"Train | Loss: {round(train_loss, 4)}, Similarity Acc: {round(train_acc, 4)}")
            print(f"Val | Loss: {round(val_loss, 4)}, Similarity Acc: {round(val_acc, 4)}")

            self.scheduler.step()
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Similarity_Acc/train", train_acc, epoch)
            writer.add_scalar("Similarity_Acc/validation", val_acc, epoch)

            if val_loss < min_val_loss:
                torch.save(self.model.state_dict(), f'{self.models_folder}/{self.name}')
                min_val_loss = val_loss

        self.model.eval()
        with torch.no_grad():
            test_loss, test_acc = self.__runner(self.test, isTrain=False)
            writer.add_text('Test Loss', str(test_loss))
            writer.add_text('Test Similarity Acc', str(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.json")
    args = parser.parse_args()

    writer = SummaryWriter()
    config = read_json(args.config)

    TrainSiamese(config).run()

    writer.flush()
    writer.close()
