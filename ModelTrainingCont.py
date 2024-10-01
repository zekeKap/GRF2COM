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
import MLDataSequencingR2
file ='C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/WBDS17walkT04.mat'
x_reshape,y_reshape,trial_y= MLDataSequencingR2.Organizing_Data(file)
# xmax,xmin=x_reshape.max(),x_reshape.min()
# x_reshape=(x_reshape-xmin)/(xmax-xmin)
model=keras.models.load_model("C:/Users/kapla/Desktop/ZEKE NUEMove/Lateral Machine Learning/RNN_Model_1.keras")
model.fit(x_reshape,y_reshape,batch_size=3,epochs=100)
keras.models.save_model(filepath='C:/Users/kapla/Desktop/ZEKE NUEMove/Lateral Machine Learning/RNN_Model_2.keras',model=model)
