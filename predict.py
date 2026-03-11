import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import argparse
from model import MNISTClassifier

def predict_digit(image_path, correctNumber):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img = Image.open(image_path)
    img = ImageOps.invert(img.convert('RGB'))

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    if correctNumber:
        fix_model(img_tensor, correctNumber)
        return

    model = MNISTClassifier()
    model.load_state_dict(torch.load("mnist_model.pth"))
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()

    print(f"--- RESULT ---")
    print(f"Digit on image: {prediction}")
    print(f"Model confidence: {confidence * 100:.2f}%")


def fix_model(img_tensor, correct_label):
    model = MNISTClassifier()
    model.load_state_dict(torch.load("mnist_model.pth"))
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    label = torch.tensor([correct_label])

    for _ in range(10):
        optimizer.zero_grad()
        output = model(img_tensor)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model trained on this image! ✅")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Розпізнавання рукописної цифри")
    parser.add_argument("image", type=str, help="Image file path (jpg/png)")
    parser.add_argument("-c" "--correctNumber", type=int, help="For training model")
    args = parser.parse_args()

    predict_digit(args.image, args.c__correctNumber)