from PIL import Image, ImageOps
import os

# Define the path to your dataset
dataset_path = './my-mnist'

# Loop over the dataset and convert the images to grayscale
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.png'):
            # Load the image and convert it to grayscale
            img_path = os.path.join(root, file)
            img = Image.open(img_path).convert('L')

            # Resize the grayscale image to 28x28 pixels
            img = img.resize((28, 28))
            
            # Invert the grayscale image
            img = ImageOps.invert(img)
            
            # Save the inverted image
            img.save(img_path)