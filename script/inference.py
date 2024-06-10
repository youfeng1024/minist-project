import torch
from torchvision import transforms
from PIL import Image
from torchvision.datasets import FashionMNIST
from model import CNN

def inference(image: Image, model_name: str):
    model_map = {
        "CNN":"checkpoints\\CNN.pth", 
        "RNN":"checkpoints\\RNN.pth", 
        "MLP":"checkpoints\\MLP.pth", 
        "RNN_ATTN":"checkpoints\\RNN_ATTN.pth"
    }
    model_path = model_map[model_name]
    
    # Load the model
    device = torch.device('cpu')
    modela = torch.load(model_path, map_location=device)
    modela = modela.to(device)
    modela.eval()  # Set the model to evaluation mode

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert the image to grayscale
        transforms.Resize((28, 28)),  # Resize the image to 28x28 pixels
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the image with the mean and standard deviation of the Fashion-MNIST dataset
    ])

    # Load and transform the image
    image = transform(image)
    image = image.unsqueeze(0)  # Add an extra dimension for batch size

    # Perform inference
    with torch.no_grad():
        output = modela(image)

    # classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    # classes = ["T恤/上衣", "裤子", "套头衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "踝靴"]
    result = FashionMNIST.classes[torch.argmax(output, dim=1)]
    return result

if __name__ == '__main__':
    import os

    # 获取当前文件所在目录的路径
    current_dir = os.path.dirname(__file__)

    # 获取上一级目录的路径
    parent_dir = os.path.dirname(current_dir)

    # 构建 路径

    image_path = os.path.join(current_dir, 'resource', 'ex1.png')

    print(image_path)
    # Load an example image
    image = Image.open(image_path)

    # Perform reasoning on the image
    print(inference(image, "CNN"))