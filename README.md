# Progressive Rock Classifier

## Install

First, create a virtual environment and install dependencies.

```sh
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, create a directory named `data` at the project root. Place the provided `Not_Progressive_Rock` and `Progressive_Rock_Songs` directories in the `data` directory.

## Preprocess

The data is not directly usable yet for model training. Run the preprocessing script to preprocess it into a usable format.

```sh
python preprocess.py
```
