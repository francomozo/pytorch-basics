import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.funcs import train_model
from src.models import CNN

# Set device
USE_CUDA = False
device = torch.device('cuda' if torch.cuda.is_available()
                      and USE_CUDA else 'cpu')


# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64

# Load Data
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

# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Needed variables for saving and loading model state
EPOCH = 1
LOSS_HISTORY = []

# Train Network
train_model(model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            curr_epoch=EPOCH,
            loss_history=LOSS_HISTORY,
            verbose=True,
            checkpoint_every=10,
            print_cuda_mem=False
            )
