from create_dataloader.change import ChangeDataloader
from Model import SiameseModel
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import read_json, calculate_accuracy, dist_filename
import argparse
import torch
import numpy as np

class TrainSiamese:
    def __init__(self, config):
        self.train, self.val, self.test = ChangeDataloader(config).create()

        self.threshold = 0.5
        self.device = config["device"]
        self.epochs = config["epochs"]
        self.name = dist_filename(config["siamese_model_name"], args.euclid)
        self.models_folder = config['models_folder']

        n_tones = len(read_json(f"{config['csv_folder']}/tones.json"))
        n_pinyins = len(read_json(f"{config['csv_folder']}/pinyins.json"))
        pretrained_model = f"{config['models_folder']}/{config['pretrained_model_name']}"

        self.model = SiameseModel(pretrained_model, n_tones, n_pinyins, self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

        self.cos_dist = nn.CosineSimilarity()
        self.cos_loss_fn = nn.CosineEmbeddingLoss()

        self.euclid_dist = nn.PairwiseDistance(p=2)

    def __euclid_loss_fn(self, embeddings: torch.Tensor, ref_embeddings: torch.Tensor, similarity: torch.Tensor, margin=1.0) -> torch.Tensor:
        """
        Euclidean Distance Loss Function defined here
        """
        dist = self.__calc_dist(embeddings, ref_embeddings)
        loss = similarity * torch.pow(dist, 2) + (1 - similarity) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        return torch.mean(loss)

    def __loss_fn(self, embeddings: torch.Tensor, ref_embeddings: torch.Tensor, similarity: torch.Tensor, margin=1.0) -> torch.Tensor:
        """
        Uses the appropriate loss function
        """
        if args.euclid:
            return self.__euclid_loss_fn(embeddings, ref_embeddings, similarity, margin)
        else:
            return self.cos_loss_fn(embeddings, ref_embeddings, similarity)

    def __calc_dist(self, embeddings: torch.Tensor, ref_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Finds the distance between 2 sets of embeddings
        """
        if args.euclid:
            return self.euclid_dist(embeddings, ref_embeddings)
        else:
            return self.cos_dist(embeddings, ref_embeddings)

    def __runner(self, dataset, isTrain=True) -> tuple:
        """
        Passes the dataset through the model
        """
        epoch_loss = 0
        y_true_similarity, y_pred_similarity = [], []

        for reference, user_input, similarity in tqdm(dataset):
            reference = reference.to(self.device)
            user_input = user_input.to(self.device)
            similarity = similarity.to(self.device)

            # Euclidean distance requires dissimilar data to be 0
            if args.euclid:
                similarity[similarity == -1] = 0

            pred_embeds, ref_embeds, _ = self.model(user_input, reference)
            loss = self.__loss_fn(pred_embeds, ref_embeds, similarity)
            
            if isTrain:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            y_true_similarity.extend(similarity)
            preds_sims = self.__calc_dist(pred_embeds, ref_embeds).detach().cpu().numpy()
            
            # Pushes the similarity to its max values to test accuracy
            if args.euclid:
                preds_sims = 1 - np.trunc(preds_sims)
            else:
                preds_sims[preds_sims > self.threshold] = 1
                preds_sims[preds_sims < self.threshold] = -1

            y_pred_similarity.extend(preds_sims.tolist())

        epoch_loss = epoch_loss / len(dataset)
        accuracy = calculate_accuracy(y_true_similarity, y_pred_similarity).item()

        return epoch_loss, accuracy


    def run(self) -> None:
        """
        Entrypoint to train the model
        """
        min_val_loss = float('inf')
        dist_type = "(Euclid)" if args.euclid else "(Cosine)"

        for epoch in range(self.epochs):
            self.model.train()
            train_loss, train_acc = self.__runner(self.train)

            self.model.eval()
            with torch.no_grad():
                val_loss, val_acc = self.__runner(self.val, isTrain=False)

            print(f"[+] {dist_type} Epoch: {epoch+1}")
            print(f"Train | Loss: {round(train_loss, 4)}, Similarity Acc: {round(train_acc, 4)}")
            print(f"Val | Loss: {round(val_loss, 4)}, Similarity Acc: {round(val_acc, 4)}")

            self.scheduler.step()
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Similarity_Acc/train", train_acc, epoch)
            writer.add_scalar("Similarity_Acc/validation", val_acc, epoch)

            # Uses the minimum val loss as best model criterion
            if val_loss < min_val_loss:
                torch.save(self.model.state_dict(), f'{self.models_folder}/{self.name}')
                min_val_loss = val_loss

        self.model.eval()
        with torch.no_grad():
            test_loss, test_acc = self.__runner(self.test, isTrain=False)
        
        print(f"[+] {dist_type} Testing Phase:")
        print(f"Loss: {test_loss}")
        print(f"Similarity Acc: {test_acc}")

        writer.add_text('Test Loss', str(test_loss))
        writer.add_text('Test Similarity Acc', str(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.json")
    parser.add_argument("--euclid", action="store_true")
    args = parser.parse_args()

    writer = SummaryWriter()
    config = read_json(args.config)

    TrainSiamese(config).run()

    writer.flush()
    writer.close()
