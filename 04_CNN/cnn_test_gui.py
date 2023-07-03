import torch
import torch.nn as nn
import torchvision.transforms as transforms

import tkinter as tk
from PIL import Image, ImageDraw


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
cnn = CNN()
cnn.load_state_dict(torch.load('cnn.pth', map_location='cpu'))
cnn.eval()


# create a window
window = tk.Tk()
window.title('My MNIST (CNN Version)')


# create a canvas
canvas = tk.Canvas(window, bg='white', width=500, height=500)
canvas.grid(row=0, column=0, columnspan=3)


# define button functions

def predict():
    # convert numpy to tensor
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)

    # predict
    with torch.no_grad():
        output = cnn(img_tensor)
        pred = output.argmax(dim=1, keepdim=True)

    # update label
    label_status.config(text='PREDICTED DIGIT: ' + str(pred.item()))


def clear():
    global img,img_draw
    
    canvas.delete('all')
    img = Image.new('L', (500, 500), color='black')
    img_draw = ImageDraw.Draw(img)    
    
    label_status.config(text='PREDICTED DIGIT: NONE')


# create buttons
button_predict = tk.Button(window, text='Predict', bg='blue', fg='white', font=('Arial', 12), command=predict)
button_predict.grid(row=1, column=0, pady=2) 

button_clear = tk.Button(window, text='Clear', bg='green', fg='white', font=('Arial', 12), command=clear)
button_clear.grid(row=1, column=1, pady=2)

button_exit = tk.Button(window, text='Exit', bg='red', fg='white', font=('Arial', 12), command=window.destroy)
button_exit.grid(row=1, column=2, pady=2)

label_status = tk.Label(window, text='Predicted Digit: NONE', bg='white', font=('Arial', 20))
label_status.grid(row=2, column=0, columnspan=3, pady=2)


# load the image
def mouse_motion(event):
    x, y = event.x, event.y
    x1, y1 = (x - 15), (y - 15)
    x2, y2 = (x + 15), (y + 15)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=30)
    img_draw.ellipse([x1, y1, x2, y2], fill='white', width=30)
canvas.bind('<B1-Motion>', mouse_motion)
img = Image.new('L', (500, 500), color='black')
img_draw = ImageDraw.Draw(img)



# start loop
window.mainloop()