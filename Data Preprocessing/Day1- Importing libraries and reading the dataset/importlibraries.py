#Importing the packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as pyplot

#Reading the csv file using pandas read_csv method
dataset = pd.read_csv('Data.csv')  #Define the proper path to Data.csv file

#Defining the independent and dependent variables
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values
