# Tone Sim Model
Predicts the pinyin and tone of a student and calculates its similarity with a reference audio.

## Training
To create the csv files:
`python initialise_dataset.py` 

To train the feature extractor:
`python train_classify.py`

 To train the siamese model:
`python train_siamese.py`

 To read training logs:
`tensorboard --logdir=runs`

## Inference
`python -W ignore inference.py -r "./examples/ao1_MV1_MP3.mp3" -i "./examples/tan2_FV1_MP3.mp3"`
