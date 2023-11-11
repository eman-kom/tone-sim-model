from utils import read_json, process_mp3
from Model import SiameseModel
from torch import nn
import torch
import argparse

def find_dist(ref_embeds, embeds):
    embeds = embeds.squeeze()
    ref_embeds = ref_embeds.squeeze()
    dist = nn.PairwiseDistance(p=2) if args.euclid else nn.CosineSimilarity(dim=0)

    return dist(embeds, ref_embeds)


def decode_one_hot(preds, mappings):
    preds = preds.squeeze()
    _, idx = torch.max(preds, 0)
    return mappings[int(idx)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config.json")
    parser.add_argument("--euclid", action="store_true")
    parser.add_argument("-i", "--user_input")
    parser.add_argument("-r", "--reference")
    args = parser.parse_args()

    config = read_json(args.config)
    ref_mp3 = process_mp3(args.reference)
    ref_mp3 = ref_mp3.unsqueeze(0)
    user_mp3 = process_mp3(args.user_input)
    user_mp3 = user_mp3.unsqueeze(0)

    swap_key_values = lambda mappings: dict(zip(mappings.values(), mappings.keys()))
    pinyins_dict = swap_key_values(read_json(f"{config['csv_folder']}/pinyins.json"))
    tones_dict = swap_key_values(read_json(f"{config['csv_folder']}/tones.json"))

    siamese_model = f"{config['models_folder']}/{config['siamese_model_name']}"
    pretrained_model = f"{config['models_folder']}/{config['pretrained_model_name']}"

    model = SiameseModel(pretrained_model, len(tones_dict), len(pinyins_dict), config["device"])
    weights = torch.load(siamese_model, map_location=config["device"], weights_only=True)
    model.load_state_dict(weights)

    model.eval()
    with torch.no_grad():
        pred_embeds, ref_embeds, pinyin_tone = model(user_mp3, ref_mp3)

    dist = find_dist(pred_embeds, ref_embeds)
    
    pinyin_preds, tone_preds = pinyin_tone
    tone = decode_one_hot(tone_preds, tones_dict)
    pinyin = decode_one_hot(pinyin_preds, pinyins_dict)
    dist_type = "(Euclid)" if args.euclid else "(Cosine)"

    print("---")
    print(f"Pinyin Prediction: {pinyin}")
    print(f"Tone Prediction  : {tone}")
    print(f"Similarity Score {dist_type}: {round(dist, 4)}")
