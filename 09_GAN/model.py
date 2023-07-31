import torch
import torch.nn as nn


# define the generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super(Generator, self).__init__()

        # define the image size
        self.img_size = img_size

        # define the latent dimension
        self.latent_dim = latent_dim

        # define the model architecture
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.img_size**2),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img
    

# define the discriminator
class Discriminator(nn.Module):
    def __init__(self, img_size=28):
        super(Discriminator, self).__init__()

        # define the model architecture
        self.model = nn.Sequential(
            nn.Linear(img_size**2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        pred = self.model(img_flat)
        return pred


# test the model
if __name__ == '__main__':
    # define the latent dimension
    latent_dim = 100

    # define the image size
    img_size = 28

    # create the generator
    gen = Generator(latent_dim, img_size)

    # create the discriminator
    disc = Discriminator(img_size)

    # generate a random noise
    noise = torch.randn(3, latent_dim)

    # generate a image
    fake_img = gen(noise)
    print(f'Generated image shape: {fake_img.shape}')

    # predict the image
    pred = disc(fake_img)

    # print the result
    print(f'Prediction: {pred}')