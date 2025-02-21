import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2


# Load TensorRT Engine
def load_trt_engine(trt_engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    return engine, context  # ✅ Removed `active_optimization_profile`


# Test inference
def test_infer():
    trt_engine, context = load_trt_engine("../yolov8n.trt")

    # Allocate Memory
    bindings = []
    device_inputs = []
    device_outputs = []
    host_outputs = []
    stream = cuda.Stream()

    for binding in trt_engine:
        size = trt.volume(trt_engine.get_tensor_shape(binding))
        dtype = trt.nptype(trt_engine.get_tensor_dtype(binding))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        if trt_engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            device_inputs.append(device_mem)
        else:
            device_outputs.append(device_mem)
            host_outputs.append(host_mem)

    # Prepare Input
    frame = cv2.imread("../test_image.jpg")
    frame_resized = cv2.resize(frame, (640, 640))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_np = np.array(frame_rgb, dtype=np.float32) / 255.0
    frame_np = np.expand_dims(frame_np.transpose(2, 0, 1), axis=0)

    cuda.memcpy_htod_async(device_inputs[0], frame_np.ravel(), stream)

    # 🔹 Correct Tensor Addressing (Ensure API compatibility)
    context.set_tensor_address(trt_engine.get_tensor_name(0), int(device_inputs[0]))
    context.set_tensor_address(trt_engine.get_tensor_name(1), int(device_outputs[0]))

    # 🔹 Run inference
    if not context.execute_async_v3(stream_handle=stream.handle):
        print("🚨 TensorRT Execution Failed!")
        return

    # Retrieve Output
    cuda.memcpy_dtoh_async(host_outputs[0], device_outputs[0], stream)
    stream.synchronize()

    print("✅ TensorRT Inference Successful!")


# Run Test
test_infer()
