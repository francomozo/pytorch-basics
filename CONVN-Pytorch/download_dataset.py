import torchvision.datasets as datasets
import torchvision.transforms as transforms

train_dataset = datasets.MNIST(
    root='dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    root='dataset/',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
