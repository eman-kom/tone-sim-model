from utils import read_json, process_mp3, dist_filename
from Model import SiameseModel
from torch import nn
import torch
import argparse
import glob

def decode_one_hot(preds, mappings):
    preds = preds.squeeze()
    _, idx = torch.max(preds, 0)
    return mappings[int(idx)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config.json")
    parser.add_argument("-f", "--folder", default="./examples")
    parser.add_argument("--euclid", action="store_true")
    args = parser.parse_args()

    cache = {}
    config = read_json(args.config)

    # load mappings
    swap_key_values = lambda mappings: dict(zip(mappings.values(), mappings.keys()))
    pinyins_dict = swap_key_values(read_json(f"{config['csv_folder']}/pinyins.json"))
    tones_dict = swap_key_values(read_json(f"{config['csv_folder']}/tones.json"))

    # load filepaths
    all_mp3s = glob.glob(f"{args.folder}/*.mp3")
    filepaths =  [filepath.replace("\\", "/").split("/")[-1] for filepath in all_mp3s]
    filepaths.sort()

    # load models
    sim_func = nn.PairwiseDistance(p=2) if args.euclid else nn.CosineSimilarity(dim=0)
    model_name = dist_filename(config['siamese_model_name'], args.euclid)
    siamese_model = f"{config['models_folder']}/{model_name}"
    pretrained_model = f"{config['models_folder']}/{config['pretrained_model_name']}"
    model = SiameseModel(pretrained_model, len(tones_dict), len(pinyins_dict), config["device"])
    weights = torch.load(siamese_model, map_location=config["device"], weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    for ref in filepaths:
        if ref not in cache:
            ref_mp3 = process_mp3(f"{args.folder}/{ref}")
            ref_mp3 = ref_mp3.unsqueeze(0)
            cache[ref] = ref_mp3

        ref_mp3 = cache[ref]

        for user_input in filepaths:
            if user_input not in cache:
                user_mp3 = process_mp3(f"{args.folder}/{user_input}")
                user_mp3 = user_mp3.unsqueeze(0)
                cache[user_input] = user_mp3

            user_mp3 = cache[user_input]

            with torch.no_grad():
                pred_embeds, ref_embeds, pinyin_tone = model(user_mp3, ref_mp3)

            ref_embeds = ref_embeds.squeeze()
            pred_embeds = pred_embeds.squeeze()
            sim = sim_func(pred_embeds, ref_embeds).item()

            if args.euclid:
                sim = 1 - sim

            pinyin_preds, tone_preds = pinyin_tone
            tone = decode_one_hot(tone_preds, tones_dict)
            pinyin = decode_one_hot(pinyin_preds, pinyins_dict)

            dist_type = "(Euclid)" if args.euclid else "(Cosine)"
            print(f"Refer: {ref}")
            print(f"Input: {user_input}")
            print(f"Pinyin Prediction: {pinyin}")
            print(f"Tone Prediction  : {tone}")
            print(f"Similarity Score {dist_type}: {round(sim, 4)}")
            print("***")
