import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_model

DATA_DIR = "data/crops"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 64
SAVE_PATH = "best_model.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, val_transform


def load_data():
    train_transform, val_transform = get_transforms()

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_dataset  = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    indices    = list(range(len(full_dataset)))
    train_set  = torch.utils.data.Subset(full_dataset, indices[:train_size])
    val_set    = torch.utils.data.Subset(val_dataset,  indices[train_size:])

    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Total: {len(full_dataset)} | Train: {train_size} | Val: {val_size}")

    class_counts = [0] * len(class_names)
    for _, label in full_dataset.samples:
        class_counts[label] += 1
    dist = ", ".join(f"{class_names[i]}={class_counts[i]}" for i in range(len(class_names)))
    print(f"Distribution: {dist}")

    total = sum(class_counts)
    class_weights = [total / c for c in class_counts]
    weight_tensor = torch.FloatTensor(class_weights)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, weight_tensor


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_weights = load_data()

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}", end="")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f" <- saved ({SAVE_PATH})")
        else:
            print()

    print(f"\nBest val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
