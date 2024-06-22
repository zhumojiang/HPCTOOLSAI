import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from model import Net
from data import train_dataset

def set_seed(seed=123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(42)


class LightningNet(pl.LightningModule):
    def __init__(self, learning_rate):
        super(LightningNet, self).__init__()
        self.model = Net()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs, labels=labels)
        loss = outputs[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        return optimizer


class TrainDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super(TrainDataModule, self).__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = train_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)


batch_size = 64
learning_rate = 0.1
epochs = 50

train_data_module = TrainDataModule(batch_size=batch_size)
lightning_model = LightningNet(learning_rate=learning_rate)


trainer = pl.Trainer(
    max_epochs=epochs,
    devices=2,  
    num_nodes=2,  
    accelerator='gpu',  
    strategy='ddp',  
    log_every_n_steps=10
)


start_time = time.time()


trainer.fit(lightning_model, train_data_module)


end_time = time.time()


total_time = end_time - start_time
print(f'Training completed in {total_time // 3600:.0f}h {total_time % 3600 // 60:.0f}m {total_time % 60:.0f}s')
