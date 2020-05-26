# -*- coding: utf-8 -*-

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Importing the dataset
dataset_train = pd.read_csv('training.csv')
dataset_test = pd.read_csv('testing.csv')

#5 class LULC system
dataset_train.iloc[dataset_train[dataset_train.label == 'forest'].index,0] = 'tree'
dataset_train.iloc[dataset_train[dataset_train.label == 'orchard'].index,0] = 'tree'
dataset_test.iloc[dataset_test[dataset_test.label == 'forest'].index,0] = 'tree'
dataset_test.iloc[dataset_test[dataset_test.label == 'orchard'].index,0] = 'tree'

#4 class LULC system (Run only after executing 5 class system labels)
dataset_train.iloc[dataset_train[dataset_train.label == 'grass'].index,0] = 'other veg.'
dataset_train.iloc[dataset_train[dataset_train.label == 'farm'].index,0] = 'other veg.'
dataset_test.iloc[dataset_test[dataset_test.label == 'grass'].index,0] = 'other veg.'
dataset_test.iloc[dataset_test[dataset_test.label == 'farm'].index,0] = 'other veg.'

# Seperating features and target variables
X_train = dataset_train.iloc[:, 2:].values
y_train = dataset_train.iloc[:, 0].values
X_test = dataset_test.iloc[:, 2:].values
y_test = dataset_test.iloc[:, 0].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.compose import ColumnTransformer

labelencoder_y = LabelEncoder()

y_train = labelencoder_y.fit_transform(y_train)
y_test = labelencoder_y.transform(y_test)

y_train = np.reshape(y_train,(-1,1))
y_test = np.reshape(y_test,(-1,1))

onehotencoder = OneHotEncoder()
y_train = onehotencoder.fit_transform(y_train).toarray()
y_test = onehotencoder.transform(y_test).toarray()


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GaussianDropout


# Initialising the RNN
classifier = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 30, return_sequences = True, input_shape = (X_train.shape[1], 1)))
classifier.add(GaussianDropout(0.4))

# Adding a second LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 30, return_sequences = True))
classifier.add(GaussianDropout(0.3))

# Adding a third LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 30, return_sequences = True))
classifier.add(GaussianDropout(0.3))

# Adding a fourth LSTM layer and some Dropout regularisation
classifier.add(LSTM(units = 30))
classifier.add(GaussianDropout(0.3))

# Adding the output layer for 4 class LULC system
classifier.add(Dense(units = 4, activation = 'softmax'))
# Adding the output layer for 5 class LULC system
#classifier.add(Dense(units = 5, activation = 'softmax'))
# Adding the output layer for 6 class LULC system
#classifier.add(Dense(units = 6, activation = 'softmax'))

# Compiling the RNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the RNN to the Training set
#history= classifier.fit(X_train, y_train,validation_data = (X_test, y_test), batch_size = 32, epochs = 15)


# Fitting the ANN to the Training set
history= classifier.fit(X_train, y_train,validation_data = (X_test, y_test), batch_size = 64, epochs = 20)

_,accr = classifier.evaluate(X_test,y_test) 

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();


y_pred = classifier.predict(X_test)
y_pred = y_pred.argmax(axis=1)
y_test1 = y_test.argmax(axis=1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred)

print((np.trace(cm)/np.sum(cm)) * 100)




