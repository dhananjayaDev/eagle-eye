# from ultralytics import YOLO
#
# # Load YOLOv8 PyTorch model
# model = YOLO("yolov8n.pt")  # Make sure your .pt file exists
#
# # Export to ONNX format
# model.export(format="onnx", dynamic=True, simplify=True)  # Create yolov8n.onnx
#
# print("✅ YOLOv8 successfully converted to ONNX!")

from ultralytics import YOLO

# Load YOLOv8 PyTorch model
model = YOLO("yolov8n.pt")  # Ensure you have the correct PyTorch model

# Export to ONNX format (with static shapes)
model.export(format="onnx", dynamic=False, opset=17, simplify=True)  # Creates yolov8n.onnx

print("✅ YOLOv8 successfully converted to ONNX with fixed shapes!")


