import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import torch.nn as nn

# load test data
test_data = datasets.MNIST(
    root='../02_dataset/data',
    train=False,
    download=True,
    transform=ToTensor()
)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
my_mnist = ImageFolder(root='../02_dataset/my-mnist', transform=transform)
my_mnist_loader = torch.utils.data.DataLoader(my_mnist, batch_size=64, shuffle=True)

# define a CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Sequential(
            nn.Linear(9216, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        logits = self.fc_2(x)
        return logits

# load the pretrained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn = CNN()
cnn.load_state_dict(torch.load('cnn.pth', map_location=device))
cnn.eval().to(device)

# test the pretrained model on MNIST test data
size = len(test_data_loader.dataset)
correct = 0

with torch.no_grad():
    for img, label in test_data_loader:
        img, label = img.to(device), label.to(device)
        pred = cnn(img)

        correct += (pred.argmax(1) == label).type(torch.float).sum().item()

correct /= size
print(f'Accuracy on MNIST: {(100*correct):>0.1f}%')

# test the pretrained model on my MNIST test data
size = len(my_mnist_loader.dataset)
correct = 0

with torch.no_grad():
    for img, label in my_mnist_loader:
        img, label = img.to(device), label.to(device)
        pred = cnn(img)

        correct += (pred.argmax(1) == label).type(torch.float).sum().item()

correct /= size
print(f'Accuracy on my MNIST: {(100*correct):>0.1f}%')