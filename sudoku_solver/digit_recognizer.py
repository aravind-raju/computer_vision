import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

batch_sz = 100
n_iters = 2500
features_train = 60000
num_epochs = n_iters / (features_train / batch_sz)
num_epochs = int(num_epochs)

class CNNModel(pl.LightningModule):

    def __init__(self, classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        out = self.softmax(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = loss_fn(y_hat, y)
        correct = (y == y_hat.argmax(axis=1)).sum()
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs, 'correct': correct, 'total': len(y)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = loss_fn(y_hat, y)
        correct = (y == y_hat.argmax(axis=1)).sum()
        logs = {'val_loss': loss}
        return {'loss': loss, 'log': logs, 'correct': correct, 'total': len(y)}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for  x in outputs])
        total = sum([x["total"] for  x in outputs])
        self.log("train_loss", avg_loss, prog_bar=True, logger=True)
        self.log("train_acc", correct/total, prog_bar=True, logger=True)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for  x in outputs])
        total = sum([x["total"] for  x in outputs])
        self.log("val_loss", avg_loss, prog_bar=True, logger=True)
        self.log("val_acc", correct/total, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor())
        loader = DataLoader(mnist_train, batch_size=batch_sz, num_workers=4)
        return loader

    def val_dataloader(self):
        mnist_val = MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor())
        return DataLoader(mnist_val, batch_size=batch_sz, num_workers=4)
