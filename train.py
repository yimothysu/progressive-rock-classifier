import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import MusicDataset

torch.manual_seed(0)

BATCH_SIZE = 64
# Proportion of data to use for training. The remaining data will be used for validation.
LEARNING_RATE = 1e-4
NUM_EPOCHS = 70

train_ds = MusicDataset("features/train", "Mel-Spectrogram")
val_ds = MusicDataset("features/valid", "Mel-Spectrogram")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)
model.fc = torch.nn.Linear(model.fc.in_features, 1)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.to("cuda")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for X, y in tqdm(train_dl):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_dl)
    print(f"Epoch {epoch + 1}, train loss: {train_loss}")

    model.eval()
    with torch.no_grad():
        val_loss = sum(
            loss_fn(model(X), y.unsqueeze(1).float()) for X, y in val_dl
        ) / len(val_dl)

    print(f"Epoch {epoch + 1}, val loss: {val_loss.item()}")
torch.save(model, "model.pt")
