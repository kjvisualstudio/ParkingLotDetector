import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import get_model

ANNOTATIONS_PATH = "./data/annotations.xml"
MODEL_PATH = "best_model.pth"
OUTPUT_PATH = "demo_output.png"
IMAGE_SIZE = 64

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

COLOR_EMPTY = (0, 255, 0)      # green
COLOR_OCCUPIED = (0, 0, 255)   # red


def load_model(device):
    model = get_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def parse_polygon(points_str):
    coords = [tuple(map(float, p.split(','))) for p in points_str.split(';')]
    return np.array(coords, dtype=np.int32)


def crop_polygon_bbox(img, pts):
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    return img[y1:y2, x1:x2]


def classify_crop(crop_bgr, model, transform, device):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        _, predicted = output.max(1)

    # ImageFolder sorts classes alphabetically: empty=0, occupied=1
    return "empty" if predicted.item() == 0 else "occupied"


def pick_demo_image(root):
    for image_elem in root.findall("image"):
        if image_elem.findall("polygon"):
            return image_elem
    return None


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    transform = get_transform()

    tree = ET.parse(ANNOTATIONS_PATH)
    root = tree.getroot()

    image_elem = pick_demo_image(root)
    if image_elem is None:
        print("No annotated images found in annotations.xml")
        return

    img_path = "data/" + image_elem.get("name")
    print(f"Running demo on: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image: {img_path}")
        return

    polygons = image_elem.findall("polygon")
    print(f"Found {len(polygons)} parking spots")

    empty_count = 0
    occupied_count = 0

    for polygon in polygons:
        pts = parse_polygon(polygon.get("points"))
        crop = crop_polygon_bbox(img, pts)

        if crop.size == 0:
            continue

        result = classify_crop(crop, model, transform, device)

        color = COLOR_EMPTY if result == "empty" else COLOR_OCCUPIED
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        if result == "empty":
            empty_count += 1
        else:
            occupied_count += 1

    summary = f"Empty: {empty_count}   Occupied: {occupied_count}"
    cv2.rectangle(img, (5, 5), (340, 40), (0, 0, 0), -1)
    cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(OUTPUT_PATH, img)
    print(f"\n{summary}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
