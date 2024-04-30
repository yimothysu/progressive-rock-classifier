import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SongDataset
from model import Model


model = Model("model.pt")
model.eval()

train_ds = SongDataset("features/train", "Mel-Spectrogram")
train_dl = DataLoader(train_ds, shuffle=False)

val_ds = SongDataset("features/valid", "Mel-Spectrogram")
val_dl = DataLoader(train_ds, shuffle=False)

test_ds = SongDataset("features/test", "Mel-Spectrogram")
test_dl = DataLoader(test_ds, shuffle=False)


def classifications(name, split_dir):
    with open(name, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Song", "True Class", "Predicted Class", "Prog Probability"])
        for i, class_dir in enumerate(["non_prog_rock", "prog_rock"]):
            for song_dir in tqdm(os.listdir(os.path.join(split_dir, class_dir))):
                features = []
                for snippet_file in os.listdir(
                        os.path.join(split_dir, class_dir, song_dir, "Mel-Spectrogram")
                ):
                    feature = (
                        torch.from_numpy(
                            np.load(
                                os.path.join(
                                    split_dir,
                                    class_dir,
                                    song_dir,
                                    "Mel-Spectrogram",
                                    snippet_file,
                                )
                            )
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to("cuda")
                    )
                    if feature.shape[-1] < 431:
                        print("Warning: snippet too short, skipping.")
                        continue
                    features.append(feature)

                model.eval()
                with torch.no_grad():
                    name = song_dir if len(song_dir) <= 40 else song_dir[:40] + "..."
                    true_class = "prog" if i == 1 else "non-prog"
                    y_pred = model.predict(features)
                    predicted_class = "prog" if y_pred else "non-prog"
                    y_pred = round(model.prob(features).item(), 4)
                    writer.writerow([name, true_class, predicted_class, y_pred])


classifications("train_classifications.csv", "features/train")
classifications("valid_classifications.csv", "features/valid")
classifications("test_classifications.csv", "features/test")
