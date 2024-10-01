import sys
import keras
import numpy
import scipy
import pandas as pd
import numpy as np
import csv
from tensorflow import _tf_uses_legacy_keras
from keras.api.models import Sequential
from keras.api.layers import SimpleRNN, Dense
from keras.api.layers import LSTM
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import MLDataSequencing
import MLDataSequencingR2
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#Doesn't work with T01 files reason yet to be determined
file='C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/WBDS17walkT02.mat'
x_reshape,y_reshape,trial_y=MLDataSequencingR2.Organizing_Data(file)
#  xmax,xmin=x_reshape.max(),x_reshape.min()
#  x_reshape=(x_reshape-xmin)/(xmax-xmin)
model=Sequential()
input=keras.Input(shape=(2,15))
model.add(input)
model.add(SimpleRNN(32,activation='tanh',return_sequences=True))
model.add(keras.layers.TimeDistributed(Dense(3)))
#optimizer= keras.optimizers.Adam(clipnorm=1.0,learning_rate=0.00001)
model.compile(optimizer= 'adam',loss='mae')
model.summary()
model.fit(x_reshape,y_reshape,batch_size=1,epochs=100)
model.save('RNN_Model_1.keras')
# test=np.vstack((Input_Data_Array[5998,0:18],Input_Data_Array[5999,0:18])).reshape((1,2,18))
# predicted_value = model.predict(test, verbose=0)
# print(predicted_value)
# print(Input_Data.loc[5999][18:])