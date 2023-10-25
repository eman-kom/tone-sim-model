import argparse
import numpy as np
import torch
import json
from tqdm import tqdm
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from Model import SiameseModel
from  pytorch_metric_learning import losses
from create_datasets.CreateChangeDataset import CreateChangeDataloader
from create_datasets.CreateClassificationDataset import CreateClassifyDataloader

class SiameseTrain:
    def __init__(self, config):
        self.train, self.val, self.test = CreateChangeDataloader(config).create()
        self.save_models_dir = config['saved_models_folder']

        self.device = config["device"]
        self.epochs = config["epochs"]
        self.criterion = losses.ContrastiveLoss()

        _,_,_, mappings = CreateClassifyDataloader(config).create()
        n_tones = len(mappings["tone_mappings"])
        n_pinyins = len(mappings["pinyin_mappings"])
        model_path = config['transfer_model']

        self.pdist = nn.PairwiseDistance(p=2)
        self.model = SiameseModel(model_path, n_tones, n_pinyins, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.9, verbose=False)

    def __contrastive_loss(self, embeddings, ref_embeddings, similarity, margin=1.0):
        dist = self.__calc_dist(embeddings, ref_embeddings)
        loss = similarity * torch.pow(dist, 2) + (1 - similarity) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        return torch.mean(loss)

    def __calc_dist(self, embeddings, ref_embeddings):
        #distances = torch.cdist(embeddings, ref_embeddings)
        #distances = torch.diagonal(distances, 0)
        distances = self.pdist(embeddings, ref_embeddings)
        return distances

    def __runner(self, dataset, isTrain=True):
        epoch_loss = 0
        y_true_similarity, y_pred_similarity = [], []

        for file1, file2, similarity in tqdm(dataset):
            file1, file2, similarity = file1.to(self.device), file2.to(self.device), similarity.to(self.device)
            pred_embeddings, ref_embeddings, _ = self.model(file1, file2, similarity)
            loss = self.__contrastive_loss(pred_embeddings, ref_embeddings, similarity)
            
            if isTrain:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            y_true_similarity.extend(similarity)
            preds_sims = self.__calc_dist(pred_embeddings, ref_embeddings).detach().cpu().numpy()
            preds_sims = 1 - np.trunc(preds_sims)
            y_pred_similarity.extend(preds_sims.tolist())

        epoch_loss = epoch_loss / len(dataset)
        correct_matches = 0
        for a, b in zip(y_true_similarity, y_pred_similarity):
            correct_matches += a == b
        similarity_acc = correct_matches.item() / len(y_pred_similarity)

        return epoch_loss, similarity_acc

    def run(self):
        min_val_loss = float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            train_loss, train_similarity_acc = self.__runner(self.train)

            self.model.eval()
            with torch.no_grad():
                val_loss, val_similarity_acc = self.__runner(self.val, isTrain=False)

            print(f"[+] Epoch: {epoch}")
            print(f"Train | Loss: {round(train_loss, 4)}, Similarity Acc: {round(train_similarity_acc, 4)}")
            print(f"Val | Loss: {round(val_loss, 4)}, Similarity Acc: {round(val_similarity_acc, 4)}")

            self.scheduler.step()
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Similarity_Acc/train", train_similarity_acc, epoch)
            writer.add_scalar("Similarity_Acc/validation", val_similarity_acc, epoch)

            if val_loss < min_val_loss:
                torch.save(self.model.state_dict(), f'{self.save_models_dir}/best_siamese_model.pth')
                val_loss = min_val_loss

        with torch.no_grad():
            _, similarity_acc = self.__runner(self.test, isTrain=False)
            writer.add_text('Test Similarity Acc', str(similarity_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", default="./config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    writer = SummaryWriter(log_dir=config['log_dir'])
    SiameseTrain(config).run()

    writer.flush()
    writer.close()
