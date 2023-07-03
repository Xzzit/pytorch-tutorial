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

# define a MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 64), # 28*28 is the size of the image
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 10), # 10 is the number of classes
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# load the pretrained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mlp = MLP()
mlp.load_state_dict(torch.load('mlp.pth'))
mlp.eval().to(device)

# test the pretrained model on MNIST test data
size = len(test_data_loader.dataset)
correct = 0

with torch.no_grad():
    for img, label in test_data_loader:
        img, label = img.to(device), label.to(device)
        pred = mlp(img)

        correct += (pred.argmax(1) == label).type(torch.float).sum().item()

correct /= size
print(f'Accuracy on MNIST: {(100*correct):>0.1f}%')

# test the pretrained model on my MNIST test data
size = len(my_mnist_loader.dataset)
correct = 0

with torch.no_grad():
    for img, label in my_mnist_loader:
        img, label = img.to(device), label.to(device)
        pred = mlp(img)

        correct += (pred.argmax(1) == label).type(torch.float).sum().item()

correct /= size
print(f'Accuracy on my MNIST: {(100*correct):>0.1f}%')