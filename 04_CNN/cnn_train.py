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

train_data_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)

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

# create a CNN model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn = CNN().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# train the model
num_epochs = 5

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}\n-------------------------------')
    for idx, (img, label) in enumerate(train_data_loader):
        size = len(train_data_loader.dataset)
        img, label = img.to(device), label.to(device)

        # compute prediction error
        pred = cnn(img)
        loss = loss_fn(pred, label)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 400 == 0:
            loss, current = loss.item(), idx*len(img)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

# save the model
torch.save(cnn.state_dict(), 'cnn.pth')
print('Saved PyTorch Model State to cnn.pth')