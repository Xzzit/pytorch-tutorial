import torch
import torch.nn as nn

"""
* The code for implementing ResNet is partially adapted from 
https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

* The original paper is available at https://arxiv.org/pdf/1512.03385.pdf

* The calcultor for the output size of convolutional layer is available at
https://madebyollin.github.io/convnet-calculator/
"""

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channel, downsample=False):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel // 4, kernel_size=1, stride=1),
                        nn.BatchNorm2d(out_channel // 4),
                        nn.ReLU(inplace=True)
                        )
        
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=3, 
                                  stride=2 if downsample else 1, padding=1),
                        nn.BatchNorm2d(out_channel // 4),
                        nn.ReLU(inplace=True)
                        )
        
        self.conv3 = nn.Sequential(
                        nn.Conv2d(out_channel // 4, out_channel, kernel_size=1, stride=1),
                        nn.BatchNorm2d(out_channel)
                        )
        
        self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel, kernel_size=1,
                                  stride=2 if downsample else 1),
                        )
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        residual = self.shortcut(x.clone())

        out = out + residual
        out = nn.ReLU(inplace=True)(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        # the first convolutional layer
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # the residual blocks
        channels = [64, 256, 512, 1024, 2048]
        self.conv2 = self._make_layer(block, channels[0], channels[1], num_blocks[0], downsample=False)
        self.conv3 = self._make_layer(block, channels[1], channels[2], num_blocks[1], downsample=True)
        self.conv4 = self._make_layer(block, channels[2], channels[3], num_blocks[2], downsample=True)
        self.conv5 = self._make_layer(block, channels[3], channels[4], num_blocks[3], downsample=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc = nn.Linear(7*7*2048, num_classes)


    def _make_layer(self, block, in_channels, out_channels, num_block, downsample):
        layers = []
        layers.append(block(in_channels, out_channels, downsample))
        for i in range(num_block - 1):
            layers.append(block(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
    

# test the model
if __name__ == '__main__':
    model = ResNet(ResidualBlock, [3, 4, 6, 3]) # ResNet50
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.size())