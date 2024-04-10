import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import MusicDataset

torch.manual_seed(0)


model = torch.load("model.pt")

model.eval()

TRAIN_SIZE = 0.8

val_ds = MusicDataset("features/valid", "Mel-Spectrogram")
val_dl = DataLoader(val_ds, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for X, y in tqdm(val_dl):
        y_pred = model(X)
        correct += (y_pred > 0).float().eq(y).sum().item()
        total += y.size(0)

print(f"Validation accuracy: {correct / total:.2f}")
