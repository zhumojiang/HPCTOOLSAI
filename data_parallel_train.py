import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np

from model import Net
from data import train_dataset

def set_seed(seed=123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed(42)



device = torch.device('cuda')
batch_size = 64
learning_rate = 0.1
epochs = 50

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Net()
model = model.to(device)
model = nn.DataParallel(model)  
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


start_time = time.time()


for epoch in range(epochs):
    epoch_start_time = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        # forward
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs, labels=labels)
        loss = outputs[0]  
        loss = loss.mean()  
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        if i % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
    epoch_end_time = time.time()
    print(f'Epoch [{epoch+1}/{epochs}] completed in {(epoch_end_time - epoch_start_time):.2f} seconds')


end_time = time.time()


total_time = end_time - start_time
print(f'Training completed in {total_time // 3600:.0f}h {total_time % 3600 // 60:.0f}m {total_time % 60:.0f}s')
