from matplotlib import pyplot as plt
import seaborn as sn
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

# TP = progressive rock, predicted as progressive rock
# TN = not progressive rock, predicted as not progressive rock
# FP = not progressive rock, predicted as progressive rock
# FN = progressive rock, predicted as not progressive rock
TP, TN, FP, FN = 0, 0, 0, 0
correct = 0
total = 0
with torch.no_grad():
    for X, y in tqdm(val_dl):
        y_pred = model(X)
        predictions = (y_pred > 0.5).float()
        TP += (predictions == 1).float().mul(y).sum().item()
        TN += (predictions == 0).float().mul(1 - y).sum().item()
        FP += (predictions == 1).float().mul(1 - y).sum().item()
        FN += (predictions == 0).float().mul(y).sum().item()
        correct += (y_pred > 0).float().eq(y).sum().item()
        total += y.size(0)

confusion_matrix = torch.tensor([[TP, FP], [FN, TN]]).to(dtype=torch.long)
ax = sn.heatmap(confusion_matrix, annot=True, fmt="d")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Progressive Rock", "Not Progressive Rock"])
ax.yaxis.set_ticklabels(["Progressive Rock", "Not Progressive Rock"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.show()
print(f"Validation accuracy: {correct / total:.2f}")
