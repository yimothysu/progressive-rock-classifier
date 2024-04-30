from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SnippetDataset
from model import Model

torch.manual_seed(0)
torch.cuda.manual_seed(0)

BATCH_SIZE = 32
# Proportion of data to use for training. The remaining data will be used for validation.
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3

train_ds = SnippetDataset("features/train", "Mel-Spectrogram")
val_ds = SnippetDataset("features/valid", "Mel-Spectrogram")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = Model()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.to("cuda")

train_losses = []
val_losses = []
print(f"Training model for {NUM_EPOCHS} epochs...")
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
    train_losses.append(train_loss)
    print(f"Epoch {epoch + 1}, train loss: {train_loss}")

    # model.eval()
    # with torch.no_grad():
    #     val_loss = sum(
    #         loss_fn(model(X), y.unsqueeze(1).float()) for X, y in val_dl
    #     ) / len(val_dl)

    # val_losses.append(val_loss.item())
    # print(f"Epoch {epoch + 1}, val loss: {val_loss.item()}")
torch.save(model, "model.pt")

print(train_losses)
print(val_losses)
plt.plot(list(range(len(train_losses))), train_losses, label="train loss")
plt.plot(list(range(len(val_losses))), val_losses, label="val loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()
