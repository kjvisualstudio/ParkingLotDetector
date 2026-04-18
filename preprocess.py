import xml.etree.ElementTree as ET
import cv2
import os

ANNOTATIONS_PATH = "./data/annotations.xml"
EMPTY_DIR = "data/crops/empty"
OCCUPIED_DIR = "data/crops/occupied"


def getMaxCoords(points):
    coords = [tuple(map(float, p.split(','))) for p in points.split(';')]
    return (int(min(coords, key=lambda p: p[0])[0]),  # x1
            int(min(coords, key=lambda p: p[1])[1]),  # y1
            int(max(coords, key=lambda p: p[0])[0]),  # x2
            int(max(coords, key=lambda p: p[1])[1]))  # y2


def createOutputDirs():
    os.makedirs(EMPTY_DIR, exist_ok=True)
    os.makedirs(OCCUPIED_DIR, exist_ok=True)
    print(f"Output dirs ready: {EMPTY_DIR}, {OCCUPIED_DIR}")


def cropAndSave(img, x1, y1, x2, y2, label, counter):
    sliced_image = img[y1:y2, x1:x2]
    if label == "free_parking_space":
        path = f'{EMPTY_DIR}/cropped{counter}.png'
        cv2.imwrite(path, sliced_image)
        print(f"  [empty]    {path}")
    else:
        path = f'{OCCUPIED_DIR}/cropped{counter}.png'
        cv2.imwrite(path, sliced_image)
        print(f"  [occupied] {path}")


def processAnnotations(root):
    counter = 0
    for image in root.findall("image"):
        img_path = "data/" + image.get("name")
        print(f"\nProcessing: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: Could not load image, skipping")
            continue

        polygons = image.findall("polygon")
        print(f"  Found {len(polygons)} polygons")

        for polygon in polygons:
            x1, y1, x2, y2 = getMaxCoords(polygon.get("points"))
            label = polygon.get("label")
            cropAndSave(img, x1, y1, x2, y2, label, counter)
            counter += 1

    return counter


def main():
    createOutputDirs()

    tree = ET.parse(ANNOTATIONS_PATH)
    root = tree.getroot()

    total = processAnnotations(root)

    empty_count = len(os.listdir(EMPTY_DIR))
    occupied_count = len(os.listdir(OCCUPIED_DIR))
    print(f"\n=== Done ===")
    print(f"Total crops saved: {total}")
    print(f"Empty:    {empty_count}")
    print(f"Occupied: {occupied_count}")


if __name__ == "__main__":
    main()
