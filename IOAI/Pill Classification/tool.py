import cv2
import os
import numpy as np
from inference_sdk import InferenceHTTPClient

image_path = "IOAI/Pill Classification/datasets/training/"

def enhance():
   kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
   DATA_PATH = os.path.join("IOAI", "Pill Classification", "datasets", "training")
   for i in range(0, 12051):
      image_name = str(i) + ".jpg"
      image_path = os.path.join(DATA_PATH, "images", image_name)
      image = cv2.imread(image_path)
      image = cv2.filter2D(image, -1, kernel)
      cv2.imwrite(image_path + 'enhanced/' + str(i) + '.jpg', image)
      print(i)

def crop():
   CLIENT = InferenceHTTPClient(
      api_url="https://detect.roboflow.com",
      api_key="tHeN54QbSKWRLCz7ZPQ9"
   )
   for i in range(0, 12051):
      image = cv2.imread(image_path + 'enhanced/' + str(i) + '.jpg')
      result = CLIENT.infer(image_path + 'enhanced/' + str(i) + '.jpg', model_id="pills-detection-s9ywn/19")
      print(result)
      if result['predictions']:
         predictions = result['predictions'][0]
         x_start = int(predictions['x'])
         x_end = x_start - int(predictions['width'])
         y_start = int(predictions['y'])
         y_end = y_start - int(predictions['height'])
         cropped_image = image[x_end:x_start, y_end:y_start]
         cv2.imwrite(image_path + 'enhanced&cropped/' + str(i) + '.jpg', cropped_image)
         print(i)

# image = cv2.imread(image_path + 'enhanced/' + str(0) + '.jpg')
# cropped = image[200:281, 249:315]
# cv2.imshow('cropped', cropped)
# cv2.waitKey(0)
crop()