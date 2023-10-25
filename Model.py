import torch
from torch import nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)

class ClassificationModel(nn.Module):
    def __init__(self, n_tones, n_pinyins):
        super().__init__()

        self.feature_extractor = FeatureExtractor()
        
        self.pinyin_prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(512, n_pinyins)
        )

        self.tone_prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(512, n_tones)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        pred_pinyin = self.pinyin_prediction(x)
        pred_tone = self.tone_prediction(x)
        return pred_pinyin, pred_tone


class SiameseModel(nn.Module):
    def __init__(self, pth_file, n_tones, n_pinyins, device):
        super().__init__()
        weights = torch.load(pth_file, map_location=device)["model_state_dict"]
        self.model = ClassificationModel(n_tones, n_pinyins)
        self.model.load_state_dict(weights)

        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_extractor = self.model.feature_extractor

        self.siam = nn.Sequential(
            self.feature_extractor,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(512, 1024)
        )

    def forward(self, x1, x2):
        preds = self.model(x1)
        x1 = self.siam(x1)
        x2 = self.siam(x2)

        return x1, x2, preds
