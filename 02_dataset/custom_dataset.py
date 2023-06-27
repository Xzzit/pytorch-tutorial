import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


# define a transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(24),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

my_mnist = ImageFolder(root='./my-mnist', transform=transform)