import sys, os, sysconfig

# 1) 复刻插件的 nvidia DLL 路径注入（GPU 加速所需）
site_dir = sysconfig.get_paths()["purelib"]
nvidia_base = os.path.join(site_dir, "nvidia")
if os.path.isdir(nvidia_base):
    for sub in os.listdir(nvidia_base):
        dll_dir = os.path.join(nvidia_base, sub, "bin")
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

# 2) 建最小 ONNX 图 (IR=11, 兼容 ORT 1.23)，Y = A + B
a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 3])
b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 3])
y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 3])
node = helper.make_node("Add", ["A", "B"], ["Y"])
g = helper.make_graph([node], "add", [a, b], [y])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], ir_version=11)
onnx.save(m, "_cuda_test.onnx")

A = np.ones((3, 3), dtype=np.float32)
B = np.ones((3, 3), dtype=np.float32) * 2

for ep in [["CUDAExecutionProvider"], ["CPUExecutionProvider"]]:
    try:
        sess = ort.InferenceSession("_cuda_test.onnx", providers=ep)
        out = sess.run(None, {"A": A, "B": B})[0]
        print(f"{ep[0]:24s} OK  -> 元素和={out[0][0]:.1f} (期望 3.0)")
    except Exception as e:
        print(f"{ep[0]:24s} FAIL -> {e}")
