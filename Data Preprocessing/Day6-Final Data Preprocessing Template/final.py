#No need to do missing data, categorical data, Feature Scaling 
#Feature Scaling is implemented by some of the algorithms, but in some cases you need to do feature scaling
#This is the final template of the data preprocessing that you will be needed to do each and every time

#Step 1 importing the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#Step2 Reading the dataset from the file
dataset = pd.read_csv('Data.csv')  #Give the path to the exact folder

#Step3 Listing the dataset as X and Y
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#Transformation of the missing data 
#First importing the library
# from sklearn.preprocessing import Imputer

# #Creating the object
# imputer = Imputer(missing_values='NaN', strategy = 'mean', axis= 0)
# imputer = imputer.fit(X[:,1:3])
# X[:,1:3] = imputer.transform(X[:,1:3])

#Categorical data encoding 
#First importing the library
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# #Encoding of X dataset
# labelencoder_X = LabelEncoder()
# X[:,0] = labelencoder_X.fit_transform(X[:,0])
# #Now doing the dummy encoding
# onehotencoder = OneHotEncoder(catgorical_features = [0])
# X = onehotencoder.fit_transform(X).toarray()

# #Encoding of the Y dataset
# labelencoder_Y = LabelEncoder()
# Y = labelencoder_Y.fit_transform(Y)

#Splitting the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

#Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
