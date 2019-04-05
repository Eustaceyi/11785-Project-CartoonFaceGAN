import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
from util.visualizer import Visualizer



class FakeDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class SimpleConv(nn.Module):
    
    def __init__(self, num_class):
        super(SimpleConv, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 4, stride=2, padding=1) # (batch, 3, 32, 32) -> (batch, 8, 16, 16)
        self.conv2 = nn.Conv2d(8, 16, 4, stride=2, padding=1) # (batch, 8, 16, 16) -> (batch, 16, 8, 8)
        self.fc = nn.Linear(8 * 8 * 16, num_class) # (batch, 16 * 8 * 8) -> (batch, num_class)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out) 
        return out

def main():
    x = np.random.randn(1000, 3, 32, 32) # random image input
    y = np.random.randint(0, 2, (1000)) # only 0 and 1
    fake_dataset = FakeDataset(x, y)
    fake_dataloader = DataLoader(fake_dataset, batch_size=64, shuffle=True, drop_last=False)
    model = SimpleConv(2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    visual= Visualizer()

    for epoch in range(10000):
        epoch_iter = 0
        total_loss = 0
        #print(epoch)
        for idx, (x, y) in enumerate(fake_dataloader):
        
            epoch_iter += 64
            optimizer.zero_grad()
            # if idx == 0:
            #     print("x shape is ", x.shape)
            #     print("y shape is ", y.shape)
            x = x.float()
            y = y.long()
            out = model.forward(x)
            _, predicted = torch.max(out.data, 1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step() 
            #print("loss is %.3f" %loss.item())
            total_loss += loss.item()

            visual.plot_loss(epoch, epoch_iter / 1000, total_loss)
            #print(y.data)
           # print(predicted)

            


if __name__ == "__main__":
    main()
