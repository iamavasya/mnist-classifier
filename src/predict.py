import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import argparse
from src.model import build_model


def get_checkpoint_path(model_name):
    return os.path.join("src", "checkpoints", f"mnist_{model_name}.pth")


def predict_digit(image_path, correctNumber, model_name='mlp', checkpoint_path=None):
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

    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(model_name)

    if correctNumber is not None:
        fix_model(img_tensor, correctNumber, model_name=model_name, checkpoint_path=checkpoint_path)
        return

    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()

    print(f"--- RESULT ---")
    print(f"Digit on image: {prediction}")
    print(f"Model confidence: {confidence * 100:.2f}%")


def fix_model(img_tensor, correct_label, model_name='mlp', checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(model_name)

    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
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

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print("Model trained on this image! ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST digit prediction")
    parser.add_argument("image", type=str, help="Image file path (jpg/png)")
    parser.add_argument("-c", "--correctNumber", type=int, help="For training model")
    parser.add_argument("--model", choices=['mlp', 'cnn'], default='mlp', help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    predict_digit(args.image, args.correctNumber, model_name=args.model, checkpoint_path=args.checkpoint)
