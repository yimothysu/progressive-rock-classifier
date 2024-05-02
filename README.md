# Progressive Rock Classifier

## Install

First, create a virtual environment and install dependencies.

```sh
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, create a directory named `data` at the project root. Place the provided `Not_Progressive_Rock`, `Progressive_Rock_Songs`, and `CAP6610SP24_test_set` directories in the `data` directory.

## Preprocess

The data is not directly usable yet for model training. Run the following preprocessing scripts to preprocess it into a usable format.

```sh
python preprocess.py
python preprocess_test_set.py
python feature_extraction.py
```

## Train

Run `python train.py` to train the model.

There are two model architectures: `CNN` and `RESNET_18` that you can choose from. `CNN` is a single convolutional layer (with BatchNorm, Max Pooling, etc.) while `RESNET_18` is an 18-layer ResNet as introduced in Kaiming He's 2015 seminal paper. The default model selection is `RESNET_18`.

## Confusion Matrices

Run `python snippet_confusion_matrix.py` to produce the snippet accuracy and confusion matrix.

Run `python song_confusion_matrix.py` to produce the song accuracy and confusion matrix.

## Other Utilities

Run `classify_to_csv.py` to generate tables displaying the classification and prediction confidence of each song.

Run `compute_statistics.py` to get the count and total runtime length of songs in a directory.
