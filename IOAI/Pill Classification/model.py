import keras
import os
import tensorflow as tf
import cv2
import json

DATA_PATH = os.path.join("IOAI", "Pill Classification", "datasets", "training")

def load_images():
   image_list = []

   for i in range(0, 12051):
      image_name = str(i) + ".jpg"
      image_path = os.path.join(DATA_PATH, "images", image_name)
      image = cv2.imread(image_path)
      image = cv2.resize(image, (224, 224))
      image_list.append(image)
      print(i)
   
   image_tensor = tf.stack(image_list)
   return image_tensor

def load_label():
   label_list = []
   label_path = os.path.join(DATA_PATH, "labels.txt")
   lf = open(label_path)
   for s in lf.readlines():
      start_index = s.index("jpg") + 4
      label_list.append(int(s[start_index:]))

   medicine_path = os.path.join(DATA_PATH, "medicine2label.json")
   mf = open(medicine_path)
   medicine_list = json.load(mf)
   medicine_list = list(medicine_list.keys())
   for i in range(0, 12051):
      label_list[i] = medicine_list[label_list[i]]
   
   return label_list

def model():
   image_tensor = load_images()
   labels = load_label()

   train_set, train_label = image_tensor[:9000], labels[:9000]
   valid_set, valid_label = image_tensor[9000:], labels[9000:]

   model = keras.Sequential([
      keras.layers.Input(shape=(224, 224, 3)),

      keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.MaxPool2D((2, 2)),

      keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.MaxPool2D((2, 2)),

      keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.MaxPool2D((2, 2)),

      keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.MaxPool2D((2, 2)),

      keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
      keras.layers.MaxPool2D((2, 2)),
      
      keras.layers.Flatten(),
      keras.layers.Dense(4096, activation="relu"),
      keras.layers.Dense(4096, activation="relu"),
      keras.layers.Dense(1000, activation="softmax"),
   ])

   model.compile(loss="sparse_categorical_crossentropy",
                 optimizer="sgd",
                 metrics=["accuracy"],)
   
   model.fit(train_set, train_label, epochs=30, validation_data=(valid_set, valid_label))
   model.save("pills.h5")

model()