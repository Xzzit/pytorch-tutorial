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

# create dataset
my_mnist = ImageFolder(root='./my-mnist', transform=transform)

# create label to idx dictionary
labels = {i: my_mnist.classes[i] for i in range(len(my_mnist.classes))}

# display images in FashionMNIST
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(my_mnist), size=(1,)).item()
    img, label = my_mnist[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# create dataloader
train_data_loader = DataLoader(my_mnist, batch_size=16, shuffle=True)
print(next(iter(train_data_loader))[0].shape)