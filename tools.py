import onnxruntime as ort
import cv2
import numpy as np

sess = ort.InferenceSession("best_model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print(input_name)
print(label_name)
image = cv2.imread('0.jpg').astype(np.float32)
image2 = cv2.imread('1.jpg').astype(np.float32)
input = np.array([image, image2])
pred = sess.run([label_name], {input_name: input})[0]
print(pred)
