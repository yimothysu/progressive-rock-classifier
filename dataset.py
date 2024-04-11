import os
from typing import Literal
import numpy as np

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

Feature = Literal["MFCC", "Mel-Spectrogram", "Beat Onset Strength", "Chromogram"]

# features/train/non_prog_rock/Song1/000.npy


class SnippetDataset(Dataset):
    def __init__(self, split_dir, feature_dir: Feature):
        self.features = []
        self.labels = []

        # Load the data
        # For each file in the directory
        for i, class_dir in enumerate(["non_prog_rock", "prog_rock"]):
            for song_dir in tqdm(os.listdir(os.path.join(split_dir, class_dir))):
                for snippet_file in tqdm(
                    os.listdir(
                        os.path.join(split_dir, class_dir, song_dir, feature_dir)
                    )
                ):
                    feature = (
                        torch.from_numpy(
                            np.load(
                                os.path.join(
                                    split_dir,
                                    class_dir,
                                    song_dir,
                                    feature_dir,
                                    snippet_file,
                                )
                            )
                        )
                        .unsqueeze(0)
                        .to("cuda")
                    )
                    if feature.shape[-1] < 431:
                        print("Warning: snippet too short, skipping.")
                        continue
                    self.features.append(feature)
                    self.labels.append(torch.tensor(i).to("cuda"))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SongDataset(Dataset):
    def __init__(self, split_dir, feature_dir: Feature):
        self.features = []
        self.labels = []

        # Load the data
        # For each file in the directory
        for i, class_dir in enumerate(["non_prog_rock", "prog_rock"]):
            for song_dir in tqdm(os.listdir(os.path.join(split_dir, class_dir))):
                self.features.append([])
                for snippet_file in tqdm(
                    os.listdir(
                        os.path.join(split_dir, class_dir, song_dir, feature_dir)
                    )
                ):
                    feature = (
                        torch.from_numpy(
                            np.load(
                                os.path.join(
                                    split_dir,
                                    class_dir,
                                    song_dir,
                                    feature_dir,
                                    snippet_file,
                                )
                            )
                        )
                        .unsqueeze(0)
                        .to("cuda")
                    )
                    if feature.shape[-1] < 431:
                        print("Warning: snippet too short, skipping.")
                        continue
                    self.features[-1].append(feature)
                self.labels.append(torch.tensor(i).to("cuda"))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
