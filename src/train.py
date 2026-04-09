import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from src.model import build_model


def get_checkpoint_path(model_name):
    return os.path.join("src", "checkpoints", f"mnist_{model_name}.pth")


def get_best_checkpoint_path(model_name):
    return os.path.join("src", "checkpoints", f"mnist_{model_name}_best.pth")

# Refactored to separate data for 3 loaders: train & val (train.py) and test (evaluate.py).
# Added validation split and random seed for reproducibility.
def build_train_val_loaders(batch_size, val_split, seed, use_pin_memory):
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(root='./data', train=True, download=True)
    total_size = len(full_dataset)

    val_size = int(total_size * val_split)
    val_size = max(1, val_size)
    train_size = total_size - val_size

    if train_size <= 0:
        raise ValueError("Validation split is too large. Keep at least one training sample.")

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(total_size, generator=generator).tolist()
    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_train, download=False)
    val_dataset = datasets.MNIST(root='./data', train=True, transform=transform_eval, download=False)

    train_loader = DataLoader(
        dataset=Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        dataset=Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader, train_size, val_size


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='MNIST model training')
    parser.add_argument('--gpu', action='store_true', help='Enable training on NVIDIA GPU')
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='mlp', help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio in range (0, 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for train/val split')
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

    if not 0 < args.val_split < 1:
        raise ValueError("--val-split must be between 0 and 1.")

    use_pin_memory = device.type == "cuda"
    train_loader, val_loader, train_size, val_size = build_train_val_loaders(
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        use_pin_memory=use_pin_memory
    )

    model = build_model(args.model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    checkpoint_path = get_checkpoint_path(args.model)
    best_checkpoint_path = get_best_checkpoint_path(args.model)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_val_acc = 0.0

    print(f"Started training with model: {args.model}")
    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            train_total += batch_size

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_checkpoint_path)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}%"
        )

    model.to("cpu")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Model trained and saved: {checkpoint_path}")
    print(f"🏆 Best validation checkpoint: {best_checkpoint_path} (val_acc={best_val_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()