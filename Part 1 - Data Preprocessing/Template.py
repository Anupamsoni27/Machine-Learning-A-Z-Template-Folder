# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:16:38 2018

@author: anupam.soni
"""

#Data Preprocessing Template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:\Anupam\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Data.csv")

x = df.iloc[:,:-1].values
y = df.iloc[:,3].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state  = 42)



#Feature Scalling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""


