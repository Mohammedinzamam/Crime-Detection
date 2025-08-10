import os
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report, accuracy_score

# Paths
model_path = './runs/detect/Normal_Compressed/weights/best.pt'  # Trained YOLOv8 model
test_images_dir = "test\images"
test_labels_dir = "test\labels"

# Load YOLOv8 model
model = YOLO(model_path)

# Function to get ground truth labels from YOLO format
def load_ground_truth(label_file):
    """
    Reads the YOLO label file and returns a list of class IDs.
    """
    if not os.path.exists(label_file) or os.path.getsize(label_file) == 0:
        return []
    with open(label_file, "r") as f:
        lines = f.readlines()
        return [int(line.strip().split()[0]) for line in lines]

# Store results
y_true = []
y_pred = []

# Iterate over test images
for filename in os.listdir(test_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(test_images_dir, filename)
        label_path = os.path.join(test_labels_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Load ground truth labels
        gt_labels = load_ground_truth(label_path)
        if not gt_labels:
            continue  # Skip images without labels

        # Run YOLO prediction
        results = model.predict(image_path, verbose=False)
        pred_labels = [int(box.cls[0]) for box in results[0].boxes]  # predicted class IDs

        # For classification report, take first label (main object in image)
        y_true.append(gt_labels[0])
        if pred_labels:
            y_pred.append(pred_labels[0])
        else:
            # If YOLO found nothing, assign a "no detection" label (-1)
            y_pred.append(-1)

# Get unique classes from dataset
classes = list(set(y_true + y_pred))
if -1 in classes:
    classes.remove(-1)  # remove "no detection" placeholder

# Compute metrics
print("\n✅ Accuracy:", accuracy_score(y_true, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_true, y_pred, labels=classes))
