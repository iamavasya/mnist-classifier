import argparse
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import build_model


def get_checkpoint_path(model_name):
    return os.path.join("src", "checkpoints", f"mnist_{model_name}.pth")


def test_model_accuracy(model_name='mlp', checkpoint_path=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(model_name)

    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    correct = 0
    total = 0

    print("Started testing 10000 images...")

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = (correct / total) * 100
    print(f"Model ({model_name}) accuracy on test data: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST model evaluation')
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp', help='Model architecture')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    args = parser.parse_args()

    test_model_accuracy(model_name=args.model, checkpoint_path=args.checkpoint)