import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import MNISTClassifier


def test_model_accuracy():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    model = MNISTClassifier()
    model.load_state_dict(torch.load("src/mnist_model.pth"))
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
    print(f"Model's accuracy on test data: {accuracy:.2f}%")


if __name__ == "__main__":
    test_model_accuracy()