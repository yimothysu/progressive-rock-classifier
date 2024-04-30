from torch import nn


class Conv_2d(nn.Module):
    def __init__(
        self, input_channels=1, output_channels=64, shape=3, pooling=2, dropout=0.1
    ):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, shape, padding=shape // 2
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(880640, 1)

    def forward(self, wav):
        out = self.conv(wav)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
