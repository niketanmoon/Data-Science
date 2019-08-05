#Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#Reading the dataset
dataset = pd.read_csv('Data.csv')  #put the original path of the data.csv file

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#Missing data transformation
#first importing the library
from sklearn.preprocessing import Imputer
#Creating a object 
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Categorical Data 
#First import the library
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
#Doing dummy encoding 
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Encoding of Y dataset
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#This is the code for today
#Splitting the dataset into training dataset and test dataset
#First importing the libraries 
from sklearn.cross_validation import train_test_split
X_train, Y_train, X_test, Y_test = train_test_split(X,Y,test_size=0.2)