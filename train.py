import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class ConvNetModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class FullyConnectedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 28 * 28)
        self.fc_out = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc_out(x)
        return x


class LitClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def get_dataloaders():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root=".", train=True, download=True, transform=transform)
    val_dataset = MNIST(root=".", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    return train_loader, val_loader


def train_conv_model(train_loader, val_loader):
    conv_model = ConvNetModule()
    model = LitClassifier(conv_model)
    trainer = pl.Trainer(max_epochs=2, accelerator="auto")
    trainer.fit(model, train_loader, val_loader)


def train_fc_model(train_loader, val_loader):
    fc_model = FullyConnectedModule()
    model = LitClassifier(fc_model)
    trainer = pl.Trainer(max_epochs=2, accelerator="auto")
    trainer.fit(model, train_loader, val_loader)


def main():
    train_loader, val_loader = get_dataloaders()
    train_conv_model(train_loader, val_loader)
    train_fc_model(train_loader, val_loader)

if __name__ == "__main__":
    main()
