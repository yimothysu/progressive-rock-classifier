import torch
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from dataset import MusicDataset

torch.manual_seed(0)


model = torch.load("model.pt")

model.eval()

TRAIN_SIZE = 0.8

dataset = MusicDataset("features", "MFCC")
num_train_instances = int(TRAIN_SIZE * len(dataset))
num_val_instances = len(dataset) - num_train_instances
train_set, val_set = random_split(
    dataset,
    [num_train_instances, num_val_instances],
)

val_dl = DataLoader(val_set, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for X, y in tqdm(val_dl):
        y_pred = model(X)
        correct += (y_pred > 0).float().eq(y).sum().item()
        total += y.size(0)

print(f"Validation accuracy: {correct / total:.2f}")
