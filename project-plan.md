# Parking Lot Detector — Project Plan
**CS3270 Intelligent Systems | Group 5 | Spring 2026**
**Team:** Kellen Andrews & JJ Araim
**Demo Due:** April 29, 2026 @ 10:30 AM

---

## Project Overview

Build a CNN-based classifier that detects whether individual parking spaces are **occupied** or **empty** from overhead parking lot images.

**Approach:**
1. Load a labeled parking lot dataset (PKLot or Kaggle)
2. Crop individual parking spots from lot images
3. Train a CNN to classify each spot as occupied or empty
4. Evaluate accuracy and visualize results on full lot images with color-coded overlays

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Language |
| PyTorch | Model training |
| OpenCV (`cv2`) | Image loading, cropping, visualization |
| torchvision | Pretrained models, transforms |
| scikit-learn | Metrics (classification report, confusion matrix) |
| matplotlib / seaborn | Plotting |
| Google Colab or local | Runtime |

Install everything:
```bash
pip install torch torchvision opencv-python scikit-learn matplotlib seaborn
```

---

## Dataset

Use the **PKLot dataset** (recommended — already labeled with spot coordinates):

> Download from Kaggle: `https://www.kaggle.com/datasets/trainingdatapro/parking-space-detection-dataset`
> OR Roboflow: `https://universe.roboflow.com/pklot-parking-detection/empty-car-parking-spot-detetection`

**Structure after download:**
```
data/
  PKLot/
    Occupied/    ← cropped spot images
    Empty/       ← cropped spot images
```

If the dataset provides full lot images + XML annotations, use `preprocess.py` (Part 1) to crop them.

---

## File Structure

```
ParkingLotDetector/
├── data/                   ← dataset goes here (not committed to git)
├── preprocess.py           ← Part 1: crop spots from full images
├── train.py                ← Part 2: train the CNN
├── evaluate.py             ← Part 3: metrics, confusion matrix
├── demo.py                 ← Part 3: visual overlay on full lot image
├── model.py                ← Part 2: model architecture definition
├── best_model.pth          ← saved after training
├── report.md               ← Part 4: written report
├── project-plan.md         ← this file
└── README.md
```

---

---

# PART 1 — Data & Preprocessing
### Assigned to: **Kellen Andrews**
**Files:** `preprocess.py`

---

### Step 1.1 — Download and organize the dataset

1. Download the PKLot dataset from Kaggle (link above)
2. Unzip into `data/PKLot/`
3. Verify the folder contains `Occupied/` and `Empty/` subdirectories with `.jpg` images
   - If not, the dataset uses XML annotations — see Step 1.2

### Step 1.2 — (If needed) Crop spots from full images

If the dataset provides full lot images with `.xml` annotation files:

```python
# preprocess.py
import os
import cv2
import xml.etree.ElementTree as ET

INPUT_DIR = "data/raw"          # full lot images + XML files
OUTPUT_OCCUPIED = "data/PKLot/Occupied"
OUTPUT_EMPTY = "data/PKLot/Empty"

os.makedirs(OUTPUT_OCCUPIED, exist_ok=True)
os.makedirs(OUTPUT_EMPTY, exist_ok=True)

for xml_file in os.listdir(INPUT_DIR):
    if not xml_file.endswith(".xml"):
        continue

    img_file = xml_file.replace(".xml", ".jpg")
    img_path = os.path.join(INPUT_DIR, img_file)
    xml_path = os.path.join(INPUT_DIR, xml_file)

    image = cv2.imread(img_path)
    if image is None:
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for i, space in enumerate(root.findall("space")):
        occupied = space.get("occupied") == "1"
        contour = space.find("contour")
        points = [(int(p.get("x")), int(p.get("y"))) for p in contour.findall("point")]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        base = xml_file.replace(".xml", f"_spot{i}.jpg")
        out_dir = OUTPUT_OCCUPIED if occupied else OUTPUT_EMPTY
        cv2.imwrite(os.path.join(out_dir, base), crop)

print("Done cropping spots.")
```

Run it:
```bash
python preprocess.py
```

### Step 1.3 — Verify the dataset

After preprocessing, check counts:
```python
import os
occupied = len(os.listdir("data/PKLot/Occupied"))
empty = len(os.listdir("data/PKLot/Empty"))
print(f"Occupied: {occupied} | Empty: {empty}")
```

You want at least a few thousand images total. If the dataset is heavily imbalanced, note it for the report.

### Step 1.4 — Write a short dataset section for the report
### Assigned to: **Kellen Andrews**

Document in `report.md`:
- Where the data came from
- How many occupied vs empty images
- Image dimensions (resize target will be 64x64 or 128x128)
- Any preprocessing decisions made

---

---

# PART 2 — Model Architecture & Training
### Assigned to: **Kellen Andrews**
**Files:** `model.py`, `train.py`, `train_results.png`

---

### Step 2.1 — Define the model (`model.py`)

Use a pretrained **ResNet-18** fine-tuned for binary classification (faster to train, better accuracy than a custom CNN):

```python
# model.py
import torch.nn as nn
from torchvision import models

def get_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    # Replace final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
```

> **Alternative:** If you want a custom CNN instead of ResNet, use this simpler architecture:
> ```python
> class ParkingCNN(nn.Module):
>     def __init__(self):
>         super().__init__()
>         self.features = nn.Sequential(
>             nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
>             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
>             nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
>         )
>         self.classifier = nn.Sequential(
>             nn.Flatten(),
>             nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.4),
>             nn.Linear(256, 2)
>         )
>     def forward(self, x):
>         return self.classifier(self.features(x))
> ```

### Step 2.2 — Write the training script (`train.py`)

```python
# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import get_model

DATA_DIR = "data/PKLot"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data loading ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print(f"Classes: {dataset.classes}")   # should be ['Empty', 'Occupied']

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# --- Model, loss, optimizer ---
model = get_model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training loop ---
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  -> Saved best model (acc: {val_acc:.4f})")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
```

Run it:
```bash
python train.py
```

### Step 2.3 — Confirm it works

- Should see loss decreasing each epoch
- Val accuracy should reach **90%+** with ResNet-18 on PKLot (dataset is relatively clean)
- `best_model.pth` should be saved

---

---

# PART 3 — Evaluation & Demo Visualization
### Assigned to: **JJ Araim**

---

### Step 3.1 — Evaluate the model (`evaluate.py`)
### Assigned to: **JJ Araim**

```python
# evaluate.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import get_model

DATA_DIR = "data/PKLot"
IMG_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32)
classes = dataset.classes  # ['Empty', 'Occupied']

model = get_model().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification report
print(classification_report(all_labels, all_preds, target_names=classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.title("Confusion Matrix — Parking Lot Detector")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")
```

### Step 3.2 — Build the visual demo (`demo.py`)
### Assigned to: **JJ Araim**

This takes a full parking lot image, uses the saved spot coordinates, runs the model on each crop, and draws a **green (empty) / red (occupied)** overlay.

```python
# demo.py
import torch
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from torchvision import transforms
from PIL import Image
from model import get_model

# --- Config ---
IMAGE_PATH = "data/raw/sample_lot.jpg"   # a full lot image
XML_PATH   = "data/raw/sample_lot.xml"   # corresponding annotation file
OUTPUT_PATH = "demo_output.jpg"
IMG_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = get_model().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

image_bgr = cv2.imread(IMAGE_PATH)
tree = ET.parse(XML_PATH)
root = tree.getroot()

occupied_count = 0
empty_count = 0

for space in root.findall("space"):
    contour = space.find("contour")
    points = [(int(p.get("x")), int(p.get("y"))) for p in contour.findall("point")]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    crop_bgr = image_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        continue

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, 1).item()

    # 0 = Empty (green), 1 = Occupied (red)
    color = (0, 255, 0) if pred == 0 else (0, 0, 255)
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(image_bgr, [pts], isClosed=True, color=color, thickness=2)

    if pred == 0:
        empty_count += 1
    else:
        occupied_count += 1

# Overlay summary text
cv2.putText(image_bgr, f"Empty: {empty_count}  Occupied: {occupied_count}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

cv2.imwrite(OUTPUT_PATH, image_bgr)
print(f"Saved demo to {OUTPUT_PATH}")
print(f"Empty: {empty_count} | Occupied: {occupied_count}")
```

Run it:
```bash
python demo.py
```
You should get `demo_output.jpg` — a full parking lot image with each space outlined green or red.

---

---

# PART 4 — Report
### Assigned to: **Both — split by section**

Write in `report.md` (or export to PDF for Moodle).
File must be submitted as part of the zip.

| Section | Owner |
|---------|-------|
| Introduction & Problem Statement | JJ |
| Dataset Description | Kellen (documented in Part 1) |
| Intelligent System Lifecycle | JJ |
| Model Architecture | Kellen |
| Training & Results | Kellen |
| Evaluation & Discussion | JJ |
| Demo Description | JJ |
| Conclusion & Ethical Considerations | Both |

**Required content per the grading rubric:**
1. How the IS lifecycle reflects in the project
2. Model description
3. Output and results (include accuracy, confusion matrix image)
4. Findings and discussion (what worked, what didn't, why)
5. How the project plan was executed (who did what, any pivots)

---

---

# PART 5 — Presentation & Submission
### Assigned to: **Both**

---

### Presentation structure (demo day April 29)

| Slide | Content | Owner |
|-------|---------|-------|
| 1 | Title, team, problem overview | Both |
| 2 | Dataset & preprocessing | Kellen |
| 3 | IS lifecycle applied to this project | JJ |
| 4 | Model architecture | Kellen |
| 5 | Training results (loss curve, accuracy) | Kellen |
| 6 | Evaluation (classification report, confusion matrix) | JJ |
| 7 | Live demo — `demo.py` on a sample lot image | JJ |
| 8 | Findings, limitations, what we'd improve | Both |
| 9 | Ethical considerations | JJ |

**Demo tip:** Run `demo.py` live during the presentation — show the color-coded lot overlay. Have a backup screenshot saved as `demo_output.jpg` in case of technical issues.

### Submission checklist (each member uploads individually to Moodle)

- [ ] Presentation slides (PDF or PPTX)
- [ ] Final report (`report.md` or PDF)
- [ ] All code files (`train.py`, `evaluate.py`, `demo.py`, `model.py`, `preprocess.py`)
- [ ] GitHub/Colab link shared with instructor
- [ ] Everything zipped as: `CS3270_Group5_AndrewsAraim_[IDs].zip`

---

## Timeline

| Date | Task |
|------|------|
| Apr 16–17 | Download dataset, run `preprocess.py`, verify data (Kellen) |
| Apr 16–18 | Write `model.py` + `train.py`, get first training run (Kellen) |
| Apr 19–20 | Tune model, hit target accuracy (Kellen) |
| Apr 20–21 | Write `evaluate.py`, generate confusion matrix (JJ) |
| Apr 21–22 | Build `demo.py`, generate lot overlay (JJ) |
| Apr 22–24 | Write report sections (both) |
| Apr 24–26 | Build presentation slides (both) |
| Apr 27 | Dry run full demo, fix any issues |
| Apr 28 | Final review, zip and upload to Moodle |
| **Apr 29** | **Demo @ 10:30 AM** |
