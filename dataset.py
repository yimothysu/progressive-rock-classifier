import os
from typing import Literal
import numpy as np

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

Feature = Literal["MFCC", "Mel-Spectrogram", "Beat Onset Strength", "Chromogram"]


class MusicDataset(Dataset):
    def __init__(self, dir, feature_type: Feature):
        self.features = []
        self.labels = []

        # Load the data
        # For each file in the directory
        for i, dir2 in enumerate(["non_prog_rock", "prog_rock"]):
            for file in tqdm(os.listdir(os.path.join(dir, dir2, feature_type))):
                feature = (
                    torch.from_numpy(
                        np.load(os.path.join(dir, dir2, feature_type, file))
                    )
                    .unsqueeze(0)
                    .to("cuda")
                )
                if feature.shape[-1] < 431:
                    print("Warning: snippet too short, skipping.")
                    continue
                self.features.append(feature)
                self.labels.append(torch.tensor(i).to("cuda"))

        # Load the feature and label
        # Append the feature and label to the respective lists

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
