import io
import base64
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms

from model import get_model

app = Flask(__name__)

IMAGE_SIZE = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
COLOR_EMPTY = (0, 255, 0)
COLOR_OCCUPIED = (0, 0, 255)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = get_model().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
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
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    return img[y1:y2, x1:x2]


def classify_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        _, predicted = output.max(1)
    return "empty" if predicted.item() == 0 else "occupied"


def load_annotations():
    tree = ET.parse("data/annotations.xml")
    root = tree.getroot()
    lookup = {}
    for image_elem in root.findall("image"):
        name = image_elem.get("name", "")  # e.g. "images/0.png"
        basename = name.split("/")[-1]      # e.g. "0.png"
        polygons = image_elem.findall("polygon")
        if polygons:
            lookup[basename] = polygons
    return lookup

ANNOTATIONS = load_annotations()


def run_inference(img_bytes, filename, xml_bytes=None):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    if xml_bytes:
        root = ET.fromstring(xml_bytes)
        polygons = root.findall("polygon")
        if not polygons:
            image_elem = root.find("image")
            if image_elem is not None:
                polygons = image_elem.findall("polygon")
    else:
        polygons = ANNOTATIONS.get(filename)

    if not polygons:
        raise ValueError(
            f"No annotations found for '{filename}'. "
            "Upload a matching annotations XML for new images."
        )

    empty_count = 0
    occupied_count = 0

    for polygon in polygons:
        pts = parse_polygon(polygon.get("points"))
        crop = crop_polygon_bbox(img, pts)
        if crop.size == 0:
            continue

        result = classify_crop(crop)
        color = COLOR_EMPTY if result == "empty" else COLOR_OCCUPIED
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        if result == "empty":
            empty_count += 1
        else:
            occupied_count += 1

    total = empty_count + occupied_count
    summary = f"Empty: {empty_count}   Occupied: {occupied_count}   Total: {total}"
    cv2.rectangle(img, (5, 5), (400, 40), (0, 0, 0), -1)
    cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)

    _, buf = cv2.imencode(".png", img)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    return {
        "image": img_b64,
        "empty": empty_count,
        "occupied": occupied_count,
        "total": total,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload a parking lot image."}), 400

    img_file = request.files["image"]
    filename = img_file.filename.split("/")[-1]
    img_bytes = img_file.read()

    xml_bytes = None
    if "annotations" in request.files and request.files["annotations"].filename:
        xml_bytes = request.files["annotations"].read()

    try:
        result = run_inference(img_bytes, filename, xml_bytes)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
