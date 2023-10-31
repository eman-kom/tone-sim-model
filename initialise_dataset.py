import argparse
from utils import read_json
from create_datasets.change import CreateChangeDataset
from create_datasets.classify import CreateClassifyDataset
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.json")
    args = parser.parse_args()
 
    config = read_json(args.config)
 
    print(" ")
    print("[+] Creating Classify Dataset")
    CreateClassifyDataset(config).save()
 
    print(" ")
    print("[+] Creating Change Dataset")
    CreateChangeDataset(config).save()
 
    print(" ")
    print("[+] Datasets Created Successfully")
    print(" ")

