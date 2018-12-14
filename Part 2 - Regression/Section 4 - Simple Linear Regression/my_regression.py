import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"C:\Anupam\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 4 - Simple Linear Regression\Salary_Data.csv")

x = df.iloc[:,:-1].values
y = df.iloc[:,1].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state  = 0)



#Feature Scalling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


#Predicting the train set results
y_pred = regressor.predict(x_test)
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Salary vs Experience (Training set")
plt.xlabel("Experence")
plt.ylabel("Salary")
plt.show()

#Predicting the Test set results
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Salary vs Experience (Training set")
plt.xlabel("Experence")
plt.ylabel("Salary")
plt.show()