# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values


# Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
# Encoding the 'country' column to numbers
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
# Encoding the 'gender' column to numbers
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

# One-hot-encoder for 'country' column
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
# Dummy variable trap
X = X[:,1:]

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Features scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Importing Keras library and packages
import keras
# Importing the Sequential module to make sequential model
from keras.models import Sequential
# Importing the type of layer we'll be using
from keras.layers import Dense
from keras.optimizers import Adam
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, activation='relu', kernel_initializer = 'uniform', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, activation='relu', kernel_initializer = 'uniform'))

# Adding the output layer
classifier.add(Dense(units = 1, activation='sigmoid', kernel_initializer = 'uniform'))

opt = Adam(lr=0.01)
# Compiling the ANN
classifier.compile(optimizer=opt, loss="binary_crossentropy", metrics = ["accuracy"])

# Train the ANN using the fit method
classifier.fit(x=X_train, y=y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_predict = classifier.predict(X_test)












