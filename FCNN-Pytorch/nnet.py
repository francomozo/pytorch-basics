# 1) Imports
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 2) Create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 3) Set device
USE_CUDA = True
device = torch.device('cuda' if torch.cuda.is_available()
                      and USE_CUDA else 'cpu')
print(f'Using device {device}')

# 4) Hyperparameters
input_size = 784  # Images of size 28x28 with one channel
hidden_size = 50
num_classes = 10
learning_rate = 0.001
batch_size = 256

# 6) Load data
train_dataset = datasets.MNIST(
    root='dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)


# Memory usage
print('Pre-training Memory Usage:')
print(f'\t Allocated: {(torch.cuda.memory_allocated(0)/1024**2):.5f} MB.')
print(f'\t Cached: {(torch.cuda.memory_reserved(0)/1024**2):.5f} MB.')

# 7) Initialize network
model = NN(
    input_size=input_size,
    hidden_size=hidden_size,
    num_classes=num_classes
).to(device)

# 8) Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)

EPOCH = 1
LOSS_HISTORY = []

# 9) Train network
while True:
    print("EPOCH:", EPOCH)
    start = time.time()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Reshape to correct shape
        # here i can use data.flatten(start_dim=1) to reshape
        data = data.reshape(data.shape[0], -1)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()  # Set gradient to zero on each batch
        loss.backward()

        # Gradient descent or Adam step
        optimizer.step()

    # Time and memory usage
    end = time.time()
    print(f'Time elapsed: {(end - start):.2f} secs.')
    print('Memory Usage:')
    print(f'\t Allocated: {(torch.cuda.memory_allocated(0)/1024**2):.5f} MB.')
    print(f'\t Cached: {(torch.cuda.memory_reserved(0)/1024**2):.5f} MB.')

    LOSS_HISTORY.append(loss.clone().detach().cpu().numpy())
    PATH = "checkpoints/model_epoch" + str(EPOCH) + ".pt"

    # if EPOCH % 5 == 0:
    #     torch.save({
    #         'epoch': EPOCH,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss_history': LOSS_HISTORY,
    #     }, PATH)

    EPOCH += 1
