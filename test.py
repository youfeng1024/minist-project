import torch
from torchvision import transforms
from PIL import Image

# Load the model
model = torch.load("checkpoints/CNN.pth", map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

# Define the image transformations
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert the image to grayscale
    transforms.Resize((28, 28)),  # Resize the image to 28x28 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image with the mean and standard deviation of the Fashion-MNIST dataset
])

# Load and transform the image
image = Image.open("image.png")
image = transform(image)
image = transforms.ToTensor()(image)  # Convert the image to a tensor
image = image.unsqueeze(0)  # Add an extra dimension for batch size

# Perform inference
with torch.no_grad():
    output = model(image)

# Print the inference results
print('finish')
print(output)