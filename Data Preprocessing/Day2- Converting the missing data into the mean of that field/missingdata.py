#Importing the libraries 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#Reading the dataset

dataset = pd.read_csv('Data.csv')  #Put the Original path of the file
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#Missing Data Handling using mean method. We are replacing the missing data with the mean of the column
#Importing the library for doing mean
from sklearn.preprocessing import Imputer

#Creating the object 
imputer = Imputer(missing_values='NaN', strategy='mean', axis = 0)
#fitting the values for imputing 
imputer = imputer.fit(X[:,1:3])  #X[:, 1:3] ':' means all the rows  and after that 1:3 means 1 and 2 column

#transforming the missing values with the mean values
X[:,1:3] = imputer.transform(X[:,1:3]) 