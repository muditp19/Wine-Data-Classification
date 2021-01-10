#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:45:27 2020

@author: m.paliwal
"""
# load the necessary libraries
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import History 
from sklearn.metrics import confusion_matrix






# Name the features of the dataset
attributes = ['Alcohol','Malic_acid','Ash','Alcalinity',
          'Magnesium','Total_phenols','Flavanoids',
          'Nonflavanoid_phenols','Proanthocyanins',
          'Color_intensity','Hue','OD280_OD315',
          'Proline']

# load the dataset with the target variable
target = 'quality'
columns = [target] + attributes
dataset = pd.read_csv('wine.data', names = columns, sep=',', header= None)


# seperating the features and labels
data = dataset[attributes]
labels = dataset[target]



# some description of the data and labels
data.info()
np.unique(labels, return_counts = True)


#one hot encoding of the labels
y_ohc = pd.get_dummies(labels, prefix='Class')

#standardizing the data 
scaler = StandardScaler()
scaler = scaler.fit(data)
data[:] = scaler.transform(data)

# splitting the dataset into training and testing set 
X_train, X_test, y_train, y_test = train_test_split(data.values, y_ohc.values, test_size=0.2, random_state=42)

# print the shapes of the created training and testing set 
print("Shapes of the train and test set : ")
print('X_train  : ',X_train.shape, '\ny_train : ',y_train.shape)
print('X_test  : ',X_test.shape, '\ny_test : ',y_test.shape)


# Defining the parameters of the Multi Layer perceptron
no_variables = X_train.shape[1]
no_classes = y_train.shape[1]

no_train = X_train.shape[0]
no_test = X_test.shape[0]

# define the number of neurons/PE in a single layer 
no_layer_input = no_variables
no_layer_hidden1 = 3
no_layer_output =  no_classes

# Model the architecture using keras Sequential Model
model=Sequential()
model.add(Dense(no_layer_hidden1, activation='relu', input_shape = (no_layer_input,))) # Single Hidden Layer
model.add(Dense(no_layer_output, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Early stopping to prevent the model from overfitting 
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, y_train, epochs=1000, verbose = 1, batch_size=1, 
					validation_split=0.2, callbacks=[es])



# plot the training and validation loss curve 
loss_values = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss_values)+1)

plt.plot(epochs, loss_values, label='Training Loss')
plt.plot(epochs, val_loss, label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



# Evaluate the model on the test data using `evaluate`
results = model.evaluate(X_test, y_test, batch_size = 1)

print('\nEvaluate on test data \n\n(loss), (accuracy) :\n{}'.format(results))

y_pred = model.predict(X_test)

# Reverting the one hot encoded labels
y_test_non_category = [ np.argmax(t) for t in y_test ]
y_predict_non_category = [ np.argmax(t) for t in y_pred ]

# Plot the confusion matrix using seaborn library
cm = confusion_matrix( y_test_non_category, y_predict_non_category)
sns.heatmap(cm, annot=True,xticklabels=['Class 1','Class 2','Class 3'], yticklabels=['Class 1','Class 2','Class 3'])

























