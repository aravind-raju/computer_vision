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

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = x.view(batch_size, -1)
        out = self.fc1(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        #self.log('val_loss', loss)
        tensorboard_logs = {'val_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

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


#model = CNNModel()
#trainer = pl.Trainer(gpus=1, max_epochs=10)
#trainer.fit(model)

#PATH = "D:/projects/computer_vision/sudoku_solver/model/digit_recognizer.pth"
#torch.save(model.state_dict(), PATH)