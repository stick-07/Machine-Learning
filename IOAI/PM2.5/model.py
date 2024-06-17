import pandas
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error
import pickle
from datetime import datetime

PATH = os.path.join("IOAI", "PM2.5")

def load_data():
   train_path = os.path.join(PATH, "datasets", "training_set.csv")
   test_path = os.path.join(PATH, "datasets", "public_test.csv")
   return pandas.read_csv(train_path), pandas.read_csv(test_path)

def prepare():
   train_set, test_set = load_data()
   train_set.dropna(subset=["PM2.5"], inplace=True)

   train_set = split_datetime(train_set)
   test_set = split_datetime(test_set)

   print(train_set)
   print(test_set)

   train_label = train_set[["PM2.5"]]
   train_set.drop(["PM2.5"], axis=1, inplace=True)

   num_pipeline = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
      ("poly", PolynomialFeatures(degree=3)),
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
   
   # model = Lasso()
   # model.fit(train_set_prepared, train_label)
   # with open('IOAI/PM2.5/ml_model.pkl', 'wb') as f:
   #    pickle.dump(model, f)

   # print(mean_absolute_percentage_error(train_label, model.predict(train_set_prepared)))

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

prepare()