import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os
import pandas
import numpy as np
from datetime import datetime

PATH = os.path.join("IOAI", "PM2.5")

def split_datetime(dataframe):
   timeset = dataframe.loc[:, 'Datetime']
   date = []
   month = []
   hour = []
   for time in timeset:
      object = datetime.strptime(time, '%Y-%m-%d  %H:%M:%S')
      date.append(object.day)
      month.append(object.month)
      hour.append(object.hour)
   dataframe.drop(['Datetime'], axis=1, inplace=True)
   dataframe['Month'] = month
   dataframe['Date'] = date
   dataframe['Hour'] = hour
   return dataframe

def load_data():
   train_path = os.path.join(PATH, "datasets", "training_set.csv")
   test_path = os.path.join(PATH, "datasets", "public_test.csv")
   return pandas.read_csv(train_path), pandas.read_csv(test_path)

def train():
   train_set, test_set = load_data()
   train_set.dropna(subset=["PM2.5"], inplace=True)

   train_set = split_datetime(train_set)
   test_set = split_datetime(test_set)

   train_label = train_set[["PM2.5"]]
   train_set.drop(["PM2.5"], axis=1, inplace=True)

   num_pipeline = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
   ])

   num_attribs = list(train_set.columns.values)
   num_attribs.remove("StationId")
   cat_attribs = ["StationId"]

   full_pipeline = ColumnTransformer([
      ("num", num_pipeline, num_attribs),
      ("cat", OneHotEncoder(), cat_attribs),
   ])

   test_set_prepared = full_pipeline.fit_transform(test_set)
   print("Done")
   train_set_prepared = full_pipeline.fit_transform(train_set)
   print("Done")

   valid_set_prepared = train_set_prepared[1500000:]
   valid_label = train_label[1500000:]
   train_set_prepared = train_set_prepared[:1500000]
   train_label = train_label[:1500000]

   zeros = np.zeros((14259, 87))
   test_set_prepared = np.hstack((test_set_prepared, zeros))
   print("Done")
   
   #2
   # model = keras.Sequential([
   #    keras.layers.Input(shape=(121,)),
   #    keras.layers.Dense(100, activation="relu"),
   #    keras.layers.Dense(75, activation="relu"),
   #    keras.layers.Dense(50, activation="relu"),
   #    keras.layers.Dense(25, activation="relu"),
   #    keras.layers.Dense(5, activation="relu"),
   #    keras.layers.Dense(1, activation="relu"),
   # ])

   #3
   model = keras.Sequential([
      keras.layers.Input(shape=(121,)),
      keras.layers.Dense(108, activation="relu"),
      keras.layers.Dense(96, activation="relu"),
      keras.layers.Dense(72, activation="relu"),
      keras.layers.Dense(36, activation="relu"),
      keras.layers.Dense(12, activation="relu"),
      keras.layers.Dense(6, activation="relu"),
      keras.layers.Dense(1, activation="relu"),
   ])

   model.compile(
      loss="mean_squared_error",
      optimizer='adam',
   )

   model.fit(train_set_prepared, train_label, epochs=15, validation_data=(valid_set_prepared, valid_label), )
   model.save("pm25network4.h5")

def model():
   train_set, test_set = load_data()
   train_set.dropna(subset=["PM2.5"], inplace=True)

   train_set = split_datetime(train_set)
   test_set = split_datetime(test_set)

   train_label = train_set[["PM2.5"]]
   train_set.drop(["PM2.5"], axis=1, inplace=True)

   num_pipeline = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
   ])

   num_attribs = list(train_set.columns.values)
   num_attribs.remove("StationId")
   cat_attribs = ["StationId"]

   full_pipeline = ColumnTransformer([
      ("num", num_pipeline, num_attribs),
      ("cat", OneHotEncoder(), cat_attribs),
   ])

   test_set_prepared = full_pipeline.fit_transform(test_set)
   print("Done")
   train_set_prepared = full_pipeline.fit_transform(train_set)
   print("Done")

   zeros = np.zeros((14259, 87))
   test_set_prepared = np.hstack((test_set_prepared, zeros))
   print("Done")
   model = keras.models.load_model(r'C:\Users\phanviethoang\Documents\Machine-Learning\IOAI\PM2.5\pm25network4.h5')
   result = model.predict(test_set_prepared)
   print(result)
   csv = pandas.DataFrame(data=result, columns=["res"])
   csv.to_csv(os.path.join(PATH, "res4.csv"))
   
train()
model()