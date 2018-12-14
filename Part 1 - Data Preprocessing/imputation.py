import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:\Anupam\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Data.csv")

x = df.iloc[:,:-1].values
y = df.iloc[:,3].values

from sklearn.preprocessing import Imputer
#      librery                     class
#Taking care of missing values
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform((x[:,1:3]))

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lableencoder_X = LabelEncoder()
x[:,0] = lableencoder_X.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

lableencoder_Y = LabelEncoder()
y = lableencoder_Y.fit_transform(y)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state  = 42)



#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)