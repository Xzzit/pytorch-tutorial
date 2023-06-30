import torch
from torchvision.transforms import ToTensor
from torchvision import datasets
import torch.nn as nn

# load MNIST dataset
training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

# create dataloader
train_data_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

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

# create a MLP model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mlp = MLP().to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# train the model
num_epochs = 5

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}\n-------------------------------')
    for idx, (img, label) in enumerate(train_data_loader):
        size = len(train_data_loader.dataset)
        img, label = img.to(device), label.to(device)

        # compute prediction error
        pred = mlp(img)
        loss = loss_fn(pred, label)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 400 == 0:
            loss, current = loss.item(), idx*len(img)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

# define test function
size = len(test_data_loader.dataset)
mlp.eval()
test_loss, correct = 0, 0

with torch.no_grad():
    for X, y in test_data_loader:
        X, y = X.to(device), y.to(device)
        pred = mlp(X)

        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

test_loss /= size
correct /= size
print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')