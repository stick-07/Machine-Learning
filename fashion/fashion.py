import pandas
import numpy as np
import keras
import tensorflow

def preprocess():
   data = keras.datasets.fashion_mnist
   (train_set, train_label), (test_set, test_label) = data.load_data()
   print(train_set.shape)
   valid_set, train_set = train_set[:5000] / 255.0, train_set[5000:] / 255.0
   valid_label, train_label = train_label[:5000], train_label[5000:]
   class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
   
   model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=[28, 28]),
      keras.layers.Dense(300, activation="relu"),
      keras.layers.Dense(100, activation="relu"),
      keras.layers.Dense(10, activation="softmax"),
   ])
   
   model.compile(loss="sparse_categorical_crossentropy",
                 optimizer="sgd",
                 metrics=["accuracy"],)
   
   model.fit(train_set, train_label, epochs=30, validation_data=(valid_set, valid_label))

   predictions = np.argmax(model.predict(test_set[:5]), axis=1)
   print(predictions)
   print(test_label[:5])

preprocess()