import torch
import torch.nn as nn
import torchvision.transforms as transforms

import tkinter as tk
from PIL import Image, ImageDraw


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
mlp = MLP()
mlp.load_state_dict(torch.load('mlp.pth', map_location='cpu'))
mlp.eval()


# create a window
window = tk.Tk()
window.title('My MNIST (MLP Version)')


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
        output = mlp(img_tensor)
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