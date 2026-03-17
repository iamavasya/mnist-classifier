import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTClassifier


def main():
    parser = argparse.ArgumentParser(description='MNIST model training')
    parser.add_argument('--gpu', action='store_true', help='Enable training on NVIDIA GPU')
    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Launching on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if args.gpu:
            print("⚠️ Warn: Argument --gpu received, but CUDA not found. Switching to CPU.")
        else:
            print("🐢 Launching on CPU. For GPU type --gpu")

    batch_size = 64
    learning_rate = 0.001
    epochs = 15

    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = MNISTClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Started training...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

    model.to("cpu")
    torch.save(model.state_dict(), "mnist_model.pth")
    print("✅ Model trained and saved!")


if __name__ == "__main__":
    main()