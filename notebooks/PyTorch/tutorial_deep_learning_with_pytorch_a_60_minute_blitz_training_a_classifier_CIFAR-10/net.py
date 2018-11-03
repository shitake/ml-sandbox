import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net,  self).__init__()
        self.conv1 = nn.Conv2d(3,  16,  5)
        self.pool = nn.MaxPool2d(2,  2)
        self.conv2 = nn.Conv2d(16,  16,  5)
        self.fc1 = nn.Linear(16 * 5 * 5,  120)
        self.fc2 = nn.Linear(120,  84)
        self.fc3 = nn.Linear(84,  10)

        def forward(self,  x):
            x = self.pool(F.relu(self.conv1(x)))  # 32x32x3 ->  28x28x6 ->14x14x6
            x = self.pool(F.relu(self.conv2(x)))  #  ->10x10x16 -> 5x5x16
            x = x.view(-1,  16 * 5 * 5)  # -> 400
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
