import keras
import keras._tf_keras
import numpy as np
import tensorflow as tf

def get_data():
   shakespeare_url = "https://homl.info/shakespeare"
   filepath = keras.utils.get_file("Shakespeare.txt", shakespeare_url)
   with open(filepath) as f:
      shakespeare_text = f.read()
   return shakespeare_text

def preprocess():
   text = get_data()
   tokenizer = keras._tf_keras.keras.preprocessing.text.Tokenizer(char_level=True)
   tokenizer.fit_on_texts([text])
   [encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1
   max_id = len(tokenizer.word_index)
   data_size = sum([x for _, x in tokenizer.word_counts.items()])
   train_size = data_size * 90 // 100
   dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
   n_steps = 100
   window_length = n_steps + 1
   dataset = dataset.window(window_length, shift=1, drop_remainder=True)
   dataset = dataset.flat_map(lambda window: window.batch(window_length))
   dataset = dataset.shuffle(10000).batch(32)
   dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
   dataset = dataset.map(lambda x_batch, y_batch: (tf.one_hot(x_batch, depth=max_id), y_batch))
   dataset = dataset.prefetch(1)

   model = keras.models.Sequential([
      keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id], dropout=0.2),
      keras.layers.GRU(128, return_sequences=True, dropout=0.2),
      keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax")),
   ])

   model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
   history = model.fit(dataset, epochs=20)

preprocess()