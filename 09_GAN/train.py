import torch

from dataloader import train_loader
from model import Generator, Discriminator

# define the hyperparameters
lr = 1e-3
latent_dim = 100
img_size = 28
num_epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create the generator
gen = Generator(latent_dim, img_size).to(device)

# create the discriminator
disc = Discriminator(img_size).to(device)

# define the loss function & optimizer
criterion = torch.nn.BCELoss()
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# define the training loop
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        pass
    pass
