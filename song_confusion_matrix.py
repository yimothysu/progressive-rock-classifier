from matplotlib import pyplot as plt
import seaborn as sn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SongDataset
from model import Model

torch.manual_seed(0)


model = Model("model.pt")
model.eval()

TRAIN_SIZE = 0.8

val_ds = SongDataset("features/test", "Mel-Spectrogram")
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
        y_pred = model.predict(X)
        TP += int(y_pred == y and y == 1)
        TN += int(y_pred == y and y == 0)
        FP += int(y_pred != y and y == 0)
        FN += int(y_pred != y and y == 1)
        correct += int(y_pred == y)
        total += 1

confusion_matrix = torch.tensor([[TP, FN], [FP, TN]]).to(dtype=torch.long)
ax = sn.heatmap(confusion_matrix, annot=True, fmt="d")
ax.set_title("Song Confusion Matrix")
ax.xaxis.set_ticklabels(["Progressive Rock", "Not Progressive Rock"])
ax.yaxis.set_ticklabels(["Progressive Rock", "Not Progressive Rock"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.show()
print(f"Validation accuracy: {correct / total:.2f}")
