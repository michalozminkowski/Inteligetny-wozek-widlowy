import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class DigitClassifier:
    def __init__(self, model_path="mnist_model.pth"):
        self.model = MNISTModel()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def predict_digit(self, path):
        img = Image.open(path).convert("L")
        img = Image.eval(img, lambda x: 255 - x)
        img = img.resize((28, 28), Image.LANCZOS)
        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)
            predicted = output.argmax(1).item()

        return predicted
