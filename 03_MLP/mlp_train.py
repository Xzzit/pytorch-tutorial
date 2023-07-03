import torch
from torchvision.transforms import ToTensor
from torchvision import datasets
import torch.nn as nn

# load MNIST dataset
training_data = datasets.MNIST(
    root='../02_dataset/data',
    train=True,
    download=True,
    transform=ToTensor()
)

train_data_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)

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
num_epochs = 20

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

# save the model
torch.save(mlp.state_dict(), 'mlp.pth')
print('Saved PyTorch Model State to mlp.pth')