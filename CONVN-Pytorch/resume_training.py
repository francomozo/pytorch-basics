import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.funcs import check_accuracy, get_last_checkpoint, train_model
from src.models import CNN

# Set device
USE_CUDA = True
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

# Load model
last_epoch = get_last_checkpoint('checkpoints/')
PATH = "checkpoints/model_epoch" + str(last_epoch) + ".pt"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
EPOCH = checkpoint['epoch']
LOSS_HISTORY = checkpoint['loss_history']

# Set model to train mode
model.train()

# Check accuracy on training data with loaded model
check_accuracy(train_loader, model, device)

# Train a little bit more
train_model(model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            curr_epoch=EPOCH,
            loss_history=LOSS_HISTORY,
            train_for=2,
            verbose=True,
            checkpoint_every=10,
            print_cuda_mem=False
            )

# Check accuracy on training data with model more trained
check_accuracy(train_loader, model, device)

# Load last checkpoint and print loss history
last_epoch = get_last_checkpoint('checkpoints/')
PATH = "checkpoints/model_epoch" + str(last_epoch) + ".pt"
checkpoint = torch.load(PATH)
LOSS_HISTORY = checkpoint['loss_history']

plt.plot(LOSS_HISTORY)
plt.ylabel('LOSS HISTORY')
plt.xlabel('EPOCH')
plt.show()
