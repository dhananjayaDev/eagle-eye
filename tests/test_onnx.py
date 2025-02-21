import onnx
import onnxruntime as ort

# Load and check ONNX model
onnx_model = onnx.load("../yolov8n.onnx")
onnx.checker.check_model(onnx_model)

# Run a simple inference test
ort_session = ort.InferenceSession("../yolov8n.onnx")
print("✅ ONNX model loaded successfully for inference!")
