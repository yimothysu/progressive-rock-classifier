import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import SnippetDataset, SongDataset
from model import Model

torch.manual_seed(0)


model = Model("model.pt")

model.eval()

TRAIN_SIZE = 0.8

val_ds = SnippetDataset("features/valid", "Mel-Spectrogram")
val_dl = DataLoader(val_ds, shuffle=False)
val_ds = SongDataset("features/valid", "Mel-Spectrogram")

correct = 0
total = 0
with torch.no_grad():
    for X, y in tqdm(val_dl):
        y_pred = model(X)
        correct += (y_pred > 0.5).float().eq(y).sum().item()
        total += y.size(0)

print(f"Snippet validation accuracy: {correct / total:.2f}")

val_dl = DataLoader(val_ds, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for X, y in tqdm(val_dl):
        y_pred = model.predict(X)
        correct += int(y_pred == y)
        total += 1

print(f"Song validation accuracy: {correct / total:.2f}")
