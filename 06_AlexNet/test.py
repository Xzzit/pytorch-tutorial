import torch

from dataloader import test_loader, classes
from model import AlexNet


# load the pretrained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AlexNet().to(device)
model.load_state_dict(torch.load('alexnet.pth', map_location=device))

# test the pretrained model on CIFAR-10 test data
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
    print('Accuracy of the network on the {} validation images: {} %'.format(10000, 100 * correct / total))