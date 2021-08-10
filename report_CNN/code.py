import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading
BITSTREAMVERA = pd.read_csv("https://raw.githubusercontent.com/CharlesColgan/MSDS-6350-6373-CODE/master/hw2/BITSTREAMVERA.csv")
CENTURY = pd.read_csv('https://raw.githubusercontent.com/sul01/MSDS-CODE/master/hw3/CENTURY.csv')
CONSOLAS = pd.read_csv("https://raw.githubusercontent.com/CharlesColgan/MSDS-6350-6373-CODE/master/hw2/CONSOLAS.csv")
EBRIMA = pd.read_csv("https://raw.githubusercontent.com/CharlesColgan/MSDS-6350-6373-CODE/master/hw2/EBRIMA.csv")
GILL = pd.read_csv('https://raw.githubusercontent.com/sul01/MSDS-CODE/master/hw3/GILL.csv')

#cleaning
drop_names = ["fontVariant","m_label","orientation","m_top","m_left","originalH","originalW","h","w"]
FONT = [BITSTREAMVERA, CENTURY, CONSOLAS, EBRIMA, GILL]; CL = []
for i in range(5):
  CL.append(FONT[i][(FONT[i]['strength']==0.4) & (FONT[i]['italic']==0)].drop(drop_names,axis=1))

#transforming
def fontToMatrix(font, df):
  for case in range(df.shape[0]):
    font[case] =  np.array(df[df.columns[3:]].iloc[case]).reshape(20,20)

FONT_mat = [] #font as 20x20 matrix
for i in range(5):
  FONT_mat.append(np.empty((CL[i].shape[0],20,20), int))
  fontToMatrix(FONT_mat[i], CL[i])

"""Examples"""

print('Bitstream\n'); plt.imshow(FONT_mat[0][8], cmap=plt.cm.binary); plt.show() 
print('Century\n'); plt.imshow(FONT_mat[1][199], cmap=plt.cm.binary); plt.show() 
print('Consolas\n'); plt.imshow(FONT_mat[2][26], cmap=plt.cm.binary); plt.show() 
print('Ebrima\n'); plt.imshow(FONT_mat[3][1387], cmap=plt.cm.binary); plt.show() 
print('Gill\n'); plt.imshow(FONT_mat[4][243], cmap=plt.cm.binary); plt.show()

"""train/test split"""

from sklearn.model_selection import train_test_split

def ypred(x, model):
  return list(map({0:'BITSTREAMVERA', 1:'CENTURY', 2:'CONSOLAS', 3:'EBRIMA', 4:'GILL'}.get, model.predict(x).argmax(axis = -1)))

dataX = np.concatenate(([FONT_mat[i] for i in range(5)])).reshape(-1,20,20,1)
dataY = np.concatenate(([CL[i]['font'] for i in range(5)]))
train, test = train_test_split(range(dataX.shape[0]), test_size = 0.2, random_state = 0)
x_train = dataX[train]; x_test = dataX[test]
y_train = dataY[train]; y_test = dataY[test]

"""#CNN

(https://www.tensorflow.org/tutorials/images/cnn)

h=200
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

#build (step 3)
model = models.Sequential()
model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(20, 20, 1)))
model.add(layers.MaxPooling2D((2, 2), strides= 2))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides= 2))
model.add(layers.Flatten())
model.add(layers.Dense(200, activation='relu')) 
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))
#model.summary()

checkpointer = ModelCheckpoint('modelh_200', monitor='val_accuracy', save_best_only=True)
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
monitor = model.fit(x_train, tf.one_hot(pd.Series(y_train,dtype='category').astype('category').cat.codes,5), epochs=100, batch_size= int(len(train)**0.5),
                    validation_data=(x_test, tf.one_hot(pd.Series(y_test, dtype='category').astype('category').cat.codes,5)), callbacks=[checkpointer])

"""Performance during learning"""

def plotCrssEntr(monitor):
  plt.plot(monitor.model.history.history['loss'], label = 'Training loss')
  plt.plot(monitor.model.history.history['val_loss'], label = 'Test loss')
  plt.xlabel('Epoch'); plt.ylabel('Loss (CrossEntr)')
  plt.legend(); plt.grid(True); plt.show()

def plotAcc(monitor):
  plt.plot(monitor.model.history.history['accuracy'], label = 'Training accuracy')
  plt.plot(monitor.model.history.history['val_accuracy'], label = 'Test accuracy')
  plt.xlabel('Epoch'); plt.ylabel('Accuracy')
  plt.legend(); plt.grid(True); plt.show()

plotCrssEntr(monitor); plotAcc(monitor)

"""Conf Matirx"""

def conf(actu, pred):
  y_actu = pd.Series(np.array(actu), name = 'Actual')
  y_pred = pd.Series(pred, name = 'Predicted')
  df_confusion = pd.crosstab(y_actu, y_pred)
  return df_confusion.divide(df_confusion.sum(axis=1),axis=0), sum(np.diag(df_confusion))/sum(np.sum(df_confusion))

trainConf, trainGlobal = conf(y_train, ypred(x_train, monitor.model))
print('Train:\n', trainConf,'\nGlobal: ', trainGlobal)

testConf, testGlobal = conf(y_test, ypred(x_test, monitor.model))
print('\nTest:\n', testConf, '\nGlobal: ', testGlobal)

"""#Repeating everything for h = 90, 150

h=90
"""

model = models.Sequential()
model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(20, 20, 1)))
model.add(layers.MaxPooling2D((2, 2), strides= 2))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides= 2))
model.add(layers.Flatten())
model.add(layers.Dense(90, activation='relu')) 
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))
#model.summary()

checkpointer = ModelCheckpoint('modelh_90', monitor='val_accuracy', save_best_only=True)
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
monitor = model.fit(x_train, tf.one_hot(pd.Series(y_train,dtype='category').astype('category').cat.codes,5), epochs=100, batch_size= int(len(train)**0.5),
                    validation_data=(x_test, tf.one_hot(pd.Series(y_test, dtype='category').astype('category').cat.codes,5)), callbacks=[checkpointer])


print('h=90:\n');plotCrssEntr(monitor); plotAcc(monitor)

trainConf, trainGlobal = conf(y_train, ypred(x_train, monitor.model))
print('Train:\n', trainConf,'\nGlobal: ', trainGlobal)

testConf, testGlobal = conf(y_test, ypred(x_test, monitor.model))
print('\nTest:\n', testConf, '\nGlobal: ', testGlobal)

"""h=150"""

model = models.Sequential()
model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(20, 20, 1)))
model.add(layers.MaxPooling2D((2, 2), strides= 2))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides= 2))
model.add(layers.Flatten())
model.add(layers.Dense(150, activation='relu')) 
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))
#model.summary()

checkpointer = ModelCheckpoint('modelh_150', monitor='val_accuracy', save_best_only=True)
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
monitor = model.fit(x_train, tf.one_hot(pd.Series(y_train,dtype='category').astype('category').cat.codes,5), epochs=100, batch_size= int(len(train)**0.5),
                    validation_data=(x_test, tf.one_hot(pd.Series(y_test, dtype='category').astype('category').cat.codes,5)), callbacks=[checkpointer])

print('h=150:\n'); plotCrssEntr(monitor); plotAcc(monitor)

trainConf, trainGlobal = conf(y_train, ypred(x_train, monitor.model))
print('Train:\n', trainConf,'\nGlobal: ', trainGlobal)

testConf, testGlobal = conf(y_test, ypred(x_test, monitor.model))
print('\nTest:\n', testConf, '\nGlobal: ', testGlobal)