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

PATH = os.path.join("IOAI", "PM2.5")

def load_data():
   train_path = os.path.join(PATH, "datasets", "training_set.csv")
   test_path = os.path.join(PATH, "datasets", "public_test.csv")
   return pandas.read_csv(train_path), pandas.read_csv(test_path)

def prepare():
   train_set, test_set = load_data()
   train_set.dropna(subset=["PM2.5"], inplace=True)

   train_label = train_set[["PM2.5"]]

   train_set.drop(["Datetime", "PM2.5"], axis=1, inplace=True)
   test_set.drop(["Datetime"], axis=1, inplace=True)

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
   print("Done")
   
   model = Lasso()
   model.fit(train_set_prepared, train_label)
   with open('IOAI/PM2.5/ml_model.pkl', 'wb') as f:
      pickle.dump(model, f)

   print(mean_absolute_percentage_error(train_label, model.predict(train_set_prepared)))

prepare()