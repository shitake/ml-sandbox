from net import Net
import os
import sys
import time
import torch.optim as optim

print(os.getcwd())
sys.path.append(os.getcwd())

# from ml-sandbox.src.utils.utils import display_formatted_time

import gc
del Training; gc.collect()

class Training:

    def __init__(self, dev):
        self.device = dev

    def train(self, lr, momentum):
        since = time.time()

        print(self.device)
        model = Net().to(self.device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=momentum)

        for epoch in range(epochs):

            running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % log_interval == (log_interval - 1):
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRunningLoss: {:.3f}".format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), running_loss / log_interval
            ))
        running_loss = 0.0

        display_formatted_time(time.time() - since)
