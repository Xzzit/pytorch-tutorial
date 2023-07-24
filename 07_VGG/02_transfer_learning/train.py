import torch
import torch.nn as nn

from dataloader import train_loader, test_loader, classes
from model import vgg16

# define the hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 20
learning_rate = 1e-3

# load the model
model = vgg16.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
total_len = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, i+1, total_len, loss.item()
            ))
            
    # Validation
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        model.train()
        print('Accuracy of the network on the {} validation images: {} %'.format(10000, 100 * correct / total))

# save the model checkpoint
torch.save(model.state_dict(), 'vgg16.pth')