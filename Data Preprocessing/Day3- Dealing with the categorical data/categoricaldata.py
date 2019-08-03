#Importing the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#Reading the dataset
dataset = pd.read_csv('Data.csv') #Put the original path of the file

#Taking the columns of features as X and independent variable as Y
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#Replacing the missing data with the mean 
#First importing the library for taking the mean
from sklearn.preprocessing import Imputer

#Creating an object
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
#fitting the columns that we need to change
imputer = imputer.fit(X[:,1:3])

#Transforming the columns with the mean values
X[:,1:3] = imputer.transform(X[:,1:3])

#Tackling the categorical data. Encoding the categorical data into numeric dataset
#First import the library for doing the encoding 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

