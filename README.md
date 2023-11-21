# Tone Sim Model
Predicts the pinyin and tone of a student and calculates its similarity with a reference audio.

Please change the settings in `config.json` to accomodate to your system. Especially the filepaths. Recommended to use absoute filepaths.

## Setting up the environment
```
conda create -n ToneSimModel
conda activate ToneSimModel
pip install -r requirements.txt
```

## Training
To initialise the dataset: `python initialise_dataset.py` 

To train the feature extractor: `python train_classify.py`

---

To train the siamese model:

With Cosine Similarity: `python train_siamese.py`

With Euclidean Distance: `python train_siamese.py --euclid`

---

To read training logs: `tensorboard --logdir=runs`

## Testing using Examples folder
With Cosine Similarity: `python -W ignore test_examples.py`

With Euclidean Distance: `python -W ignore test_examples.py --euclid`

## Inference
With Cosine Similarity: `python -W ignore inference.py -r "./examples/ao1_MV1_MP3.mp3" -i "./examples/tan2_FV1_MP3.mp3"`

With Euclidean Distance: `python -W ignore inference.py -r "./examples/ao1_MV1_MP3.mp3" -i "./examples/tan2_FV1_MP3.mp3" --euclid`
