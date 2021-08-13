import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# architectures
class Discriminator(nn.Module):
    # It is the inspector, or the policeman
    def __init__(self, in_features):
        super().__init__()
        
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    # It is the forger
    # z_dim : dimension of noise
    def __init__(self, z_dim, img_dim):
        super().__init__()
        
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
        nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)
    
# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {device}')
lr = 3e-4 # Andrej Karpathy's tweet
z_dim = 64 # can try 128, 256, etc
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 500

# Initializations
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device) # to see how it changes through time


transforms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
])
dataset_path = '/'.join(os.getcwd().split('/')[:-1]) + '/dataset/' # uglycode
dataset = datasets.MNIST(root=dataset_path, transform=transforms, download=False)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss() # implements the form of the GANs loss. The game is to minimize this loss
                         # for both networks. Even though the theoretical losses are to be maximized
                         # nn.BCELoss has a minus "-" at the beggining, so it is the same as minimizing
                         # -"the loss functions"

writer_fake = SummaryWriter(f"runs/500epochs/fake")
writer_real = SummaryWriter(f"runs/500epochs/real")
step = 0


for epoch in range(num_epochs):
    disc_epoch_loss = []
    gen_epoch_loss = []

    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)        
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device) # Gaussian distribution
        fake = gen(noise) # generate fake images
        
        # Part of the loss function for real images
        disc_real = disc(real).view(-1) # output of the Discriminator on real image
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # torch.ones is the y_n in the
                                                                      # doc of BCELoss *****
        # Same but with fake images output from the generator
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # *****
        # ****: for lossD_real we want the first term (y_n * log(x_n)), whether
        #       in lossD_fake we want the second ((1 - y_n * log(1 - x_n))), so this explains
        #       the why in torch.ones_like and torch.zeros_like
        lossD = (lossD_real + lossD_fake) / 2
        disc_epoch_loss.append(lossD.item())
        disc.zero_grad()
        lossD.backward(retain_graph=True) # retain_graph=True because I want to reuse fake=gen(noise)
                                          # in the next section
        opt_disc.step()
        
        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen_epoch_loss.append(lossG.item())
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # Tensorboard code
        if batch_idx == 0:
            disc_loss = sum(disc_epoch_loss)/len(disc_epoch_loss)
            gen_loss = sum(gen_epoch_loss)/len(gen_epoch_loss)
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {disc_loss:.4f}, loss G: {gen_loss:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
