# Parking Lot Detector

A computer vision tool that classifies parking spots as empty or occupied using a fine-tuned ResNet-18 model.

## How It Works

1. Parking lot images are annotated with polygon regions marking individual spots
2. The preprocessing step crops each spot into its own image and labels it (empty/occupied)
3. A ResNet-18 model (pretrained on ImageNet) is fine-tuned on these cropped images
4. At inference time, the model classifies each annotated spot and overlays the results on the original image

## Project Structure

- `preprocess.py` — Parses `annotations.xml` and crops individual parking spots into `data/crops/`
- `model.py` — ResNet-18 with a frozen backbone and trainable classification head
- `train.py` — Trains the model using a 70/15/15 train/val/test split
- `evaluate.py` — Evaluates the trained model on the held-out test set
- `demo.py` — Runs inference on an annotated image and saves the result
- `app.py` — Flask web app for uploading images and viewing results

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision flask opencv-python pillow matplotlib seaborn scikit-learn
```

## Usage

**Preprocess the data:**
```
python preprocess.py
```

**Train the model:**
```
python train.py
```

**Evaluate on the test set:**
```
python evaluate.py
```

**Run the demo:**
```
python demo.py
```

**Start the web app:**
```
python app.py
```
Then open http://127.0.0.1:5000 in a browser.

## Notes

- The model requires annotation files (polygon coordinates) to identify parking spot locations in an image
- Images from `annotations.xml` can be used in the web app without uploading a separate XML file
- New/unseen parking lot images require their own annotations XML
