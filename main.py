import cv2
from PIL import Image

from ultralytics import YOLO
# Create a new YOLO model from scratch
model = YOLO("yolo11n.yaml")
# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")
# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="/mnt/e/Voxel51/dataset1/data.yaml", epochs=3)
# Evaluate the model's performance on the validation set
results = model.val()
# Perform object detection on an image using the model
results = model("https://cdn.mos.cms.futurecdn.net/HJ4M2Z7kqbhqDrV6xXa7s-700-80.jpg")
# Export the model to ONNX format
success = model.export(format="onnx")


# from PIL
im1 = Image.open("valid3.jpg")
results = model.predict(source=im1, save=True)  # save plotted images

