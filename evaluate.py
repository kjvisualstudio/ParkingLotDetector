import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from model import get_model

DATA_DIR = "data/crops"
MODEL_PATH = "best_model.pth"
IMAGE_SIZE = 64
BATCH_SIZE = 32
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(device):
    model = get_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def load_test_set():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    n = len(full_dataset)
    train_size = int(TRAIN_RATIO * n)
    val_size = int(VAL_RATIO * n)

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n, generator=generator).tolist()

    test_indices = indices[train_size + val_size:]
    test_set = torch.utils.data.Subset(full_dataset, test_indices)

    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return loader, full_dataset.classes


def run_inference(model, loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Saved: confusion_matrix.png")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    loader, class_names = load_test_set()
    print(f"Classes: {class_names} | Test samples: {len(loader.dataset)}")

    labels, preds = run_inference(model, loader, device)

    print("\n=== Classification Report (Test Set) ===")
    print(classification_report(labels, preds, target_names=class_names))

    save_confusion_matrix(labels, preds, class_names)


if __name__ == "__main__":
    main()
