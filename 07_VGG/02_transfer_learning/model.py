import torch
import torch.nn as nn
from torchvision import models

# load pretrained vgg16 model
vgg16 = models.vgg16(weights='IMAGENET1K_V1')

# freeze the parameters
for param in vgg16.parameters():
    param.requires_grad = False

# replace the last fully-connected layer
vgg16.classifier[6] = nn.Linear(4096, 10)

# replace the whole classifier
# vgg16.classifier = nn.Sequential(
#     nn.Linear(25088, 4096),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(4096, 4096),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(4096, 10)
# )

# test the model
if __name__ == '__main__':

    # test the model on a random tensor
    img = torch.randn(1, 3, 224, 224)
    output = vgg16(img)
    print(output.shape)