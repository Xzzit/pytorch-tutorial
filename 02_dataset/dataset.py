import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# download training & testing dataset
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

# create label to idx dictionary
labels = {i: training_data.classes[i] for i in range(len(training_data.classes))}

# display images in FashionMNIST
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# create dataloader
train_data_loader = DataLoader(training_data, batch_size=16, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=16, shuffle=True)
print(next(iter(train_data_loader))[0].shape)