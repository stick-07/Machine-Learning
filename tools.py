import PIL.Image
import cv2

image = cv2.imread("0.jpg")
print(image[0])
image = cv2.resize(image, (227, 227))
print(image[0])
cv2.imwrite("1.jpg", image)
