"""
Training of DCGAN network on MNIST dataset with Discriminator
and Generator imported from models.py
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, Generator, initialize_weights

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#LEARNING_RATE = 3e-4  # could also use two lrs, one for gen and one for disc
#BATCH_SIZE = 512
IMAGE_SIZE = 64
CHANNELS_IMG = 1
#NOISE_DIM = 100
NUM_EPOCHS = 10
# next is 64 for both to match what they did in the paper
#FEATURES_DISC = 64
#FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)] # works for any number
                                                            # of channels of images
        ),
    ]
)

dataset_path = '/'.join(os.getcwd().split('/')[:-1]) + '/dataset/' # uglycode
# If you train on MNIST, remember to set channels_img to 1

writer_real = SummaryWriter(f"runs_gs2/real")
writer_fake = SummaryWriter(f"runs_gs2/fake")
step = 0


lrs = [3e-3]
bsizes = [1024]
noise_dims = [128]
features = [64]

hyperparams = [(lr, bsize, noise_dim, ft) for lr in lrs 
                                                for bsize in bsizes 
                                                for noise_dim in noise_dims 
                                                for ft in features]
# lr, bsize, noise dim
for (lr, bsize, noise_dim, ft) in hyperparams:
    FEATURES_DISC = ft
    FEATURES_GEN = ft
    
    dataset = datasets.MNIST(root=dataset_path, train=True, transform=transforms,
                       download=True)
    dataloader = DataLoader(dataset, batch_size=bsize, shuffle=True)
    
    gen = Generator(noise_dim, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999)) #beta1 from paper
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, noise_dim, 1, 1).to(device)

    gen.train()
    disc.train()
    
    writer = SummaryWriter(
            f"runs_gs2/hparams/lr {lr} bsize {bsize} noise_dim {noise_dim} ft {ft}"
        )
    print() 
    for epoch in range(NUM_EPOCHS):
        losses_epoch_disc = []
        losses_epoch_gen = []
        # same as the fc example (SimpleGAN)
        for batch_idx, (real, _) in enumerate(dataloader):

            real = real.to(device)
            noise = torch.randn(bsize, noise_dim, 1, 1).to(device)
            fake = gen(noise)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1) # fake.detach does the same asi setting
                                                        # retain_graph=True in loss_disc.backward()
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            # for gs
            losses_epoch_disc.append(loss_disc.item())
            
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            # for gs
            losses_epoch_gen.append(loss_gen.item())
            
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} lr {lr}, bsize {bsize}, noise_dim {noise_dim}, ft {ft} \
                      Loss D: {loss_disc:.6f}, loss G: {loss_gen:.6f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1


        if (epoch + 1) == NUM_EPOCHS:       
        	writer.add_hparams(
            	    {"lr": lr, "bsize": bsize, "noise_dim": noise_dim, "ft": ft},
                	{
                    	"loss_D": sum(losses_epoch_disc) / len(losses_epoch_disc),
                    	"loss_G": sum(losses_epoch_gen) / len(losses_epoch_gen),
                	},
            )
