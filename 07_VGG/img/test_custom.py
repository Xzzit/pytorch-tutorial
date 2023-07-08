import torch
from torchvision import models
from torchvision import transforms

from PIL import Image
import json
import argparse


# define parser
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='car.jpg', help='test image path')
args = parser.parse_args()

# load pretrained vgg16 model
vgg16 = models.vgg16(weights='IMAGENET1K_V1')
vgg16.eval()

# define transform
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# load image
img = Image.open(args.img_path)
img = transform(img)
img = img.unsqueeze(0)

# load class
class_idx = json.load(open("imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# inference
output = vgg16(img)
pred = torch.argmax(output, dim=1).item()
print(idx2label[pred])