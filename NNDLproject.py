# -*- coding: utf-8 -*-

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ann_visualizer.visualize import ann_viz
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = StandardScaler()
#sc = MinMaxScaler(feature_range=(0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, GaussianDropout
from keras import regularizers
from keras.optimizers import Adam

# For six class LULC system
'''
class_weight = { 0 : 1,
                 1 : 1,
                 2 : 1,
                 3 : 1,
                 4 : 5,
                 5 : 1
                }
'''

# For five class LULC system

'''
class_weight = { 0 : 1,
                 1 : 5,
                 2 : 1,
                 3 : 1,
                 4 : 2
                }
'''

# For four class LULC system
class_weight = { 0 : 2,
                 1 : 3,
                 2 : 2,
                 3 : 3,
                }

#For 4 class LULC system
#adam = Adam(lr=0.0009)
#For 5 class LULC system
adam = Adam(lr=0.001)
#For 6 class LULC system
#adam = Adam(lr=0.0008)
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 20, activation = 'softplus', kernel_initializer = 'uniform', input_dim = 27))
classifier.add(GaussianDropout(rate = 0.3))

# Adding the second hidden layer
classifier.add(Dense(units = 20, activation = 'softplus', kernel_initializer = 'uniform'))
classifier.add(GaussianDropout(rate = 0.3))

# Adding the output layer
#classifier.add(Dense(units = 6,  kernel_initializer = 'uniform', activation = 'softmax'))
#classifier.add(Dense(units = 5,  kernel_initializer = 'uniform', activation = 'softmax'))
classifier.add(Dense(units = 4,  kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
#history= classifier.fit(X_train, y_train,validation_data = (X_test, y_test), batch_size = 16, epochs = 200, class_weight=class_weight)
history= classifier.fit(X_train, y_train,validation_data = (X_test, y_test), batch_size = 32, epochs = 200, class_weight=class_weight)


_,accr = classifier.evaluate(X_test,y_test)




plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();


# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dropout, GaussianNoise
from keras import regularizers
from keras.layers import Dense


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 20, activation = 'softplus', kernel_initializer = 'uniform', input_dim = 27))
    classifier.add(GaussianDropout(rate = 0.3))
    classifier.add(Dense(units = 20, activation = 'softplus', kernel_initializer = 'uniform'))
    classifier.add(GaussianDropout(rate = 0.3))
    classifier.add(Dense(units = 5,  kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs = 200)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

history = classifier.fit(X_train, y_train)

_,accr = classifier.evaluate(X_test,y_test)

y_pred = classifier.predict(X_test)
y_pred = y_pred.argmax(axis=1)
y_test1 = y_test.argmax(axis=1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred)

print((np.trace(cm)/np.sum(cm)) * 100)




