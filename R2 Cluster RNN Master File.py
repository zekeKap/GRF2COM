# I need to change Lateral Machine Learning into a function and Call it with all the arguments necessary
# everything else is a function basically
# I need to make a directory of all the usable .mat files
#
import os
import pandas as pd
import MLDataSequencingR3
import keras
from keras.api.layers import SimpleRNN, Dense, LSTM
from keras.api.models import Sequential
import keras.api.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import Data_Adjustment


def absError(y_true, y_pred):
    f_y_pred = keras.ops.numpy.ravel(y_pred)
    f_y_true = keras.ops.numpy.ravel(y_true)
    e = tf.math.reduce_sum(f_y_pred - f_y_true)
    return abs(e)con

unit = int(256)
batch = int(1000)
reps=int(2)
count=0
DataSet = os.listdir("/scratch/kaplan.ez/WalkingDataSet")
Path_Dataset = "/scratch/kaplan.ez/WalkingDataSet/"
DatasetInfoFile = "WBDSinfo.csv"
DatasetInfo = pd.read_csv(DatasetInfoFile)
print(DatasetInfo)
Subject = []
for i in range(len(DataSet)):
    subject = DataSet[i][4:6]
    Subject.append(int(subject))
print(Subject)
model=Sequential()
model.add(SimpleRNN(unit,activation='tanh',return_sequences=False,input_shape=(1,13),dropout=0.05))
model.add(Dense(int(unit*2),activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dense(units=unit,activation='relu'))
model.add(Dense(3,activation='linear'))
#optimizer= keras.optimizers.Adam(clipnorm=1.0,learning_rate=0.00001)
model.compile(optimizer= 'adam',loss='mse',metrics=['root_mean_squared_error']) #RMSE solves our scaling problem
model.summary()
# I now need to have the subject dictate the automated part of the Height and Weight parts
# the question is how should I set this up I can do the hard code with the for loop
Xtraining=[]
Ytraining=[]
training_order=np.linspace(0,len(DataSet)-1,len(DataSet))
for k in range(len(DataSet)):
  print(k/len(DataSet))
  for j in range(len(DatasetInfo) - 1):
    if Subject[k] == DatasetInfo["Subject"][j]:
      Height = float(DatasetInfo["Height"][j] / 100)
      Mass = float(DatasetInfo["Mass"][j])
      file = Path_Dataset + DataSet[k]
      x_reshape, y_reshape, trial_y = MLDataSequencingR3.Organizing_Data(file, float(Height), float(Mass))  # Add Height and Mass
      x_reshape = x_reshape.astype(np.float64)
      y_reshape = y_reshape.astype(np.float64)
      Xtraining.append(x_reshape)
      Ytraining.append(y_reshape)
      print(file)
      break
  print(np.array_equal(Xtraining[k],x_reshape))
  
for rep in range(1,reps):
    np.random.shuffle(training_order)
    for z in range(len(Xtraining)):
        index=int(training_order[z])
        count=count+1
        percent=float((count/(len(Xtraining)*reps))*100)
        x_reshape=Xtraining[index]
        y_reshape=Ytraining[index]
        x = Data_Adjustment.COP_adjustment(x_reshape)
        x, y = Data_Adjustment.Data_Normalization(x, y_reshape)
        x = Data_Adjustment.reshape(x)
        y = Data_Adjustment.reshapeoutput(y_reshape, 5)
        model.reset_metrics()
        model.fit(x, y, batch_size=batch, epochs=50,shuffle=True)  # larger batch size the better it is
        print(percent)
    model.save('RNN_Model_Multi_Set.keras')
capturePredictedValues = model(x,training=False)
# cogx=(capturePredictedValues*(y_max-y_min))+(y_min)
plt.plot(capturePredictedValues[:, 0])
plt.plot(y[:, 0])
# plt.plot(error,'o')
plt.legend(["predict", "GT"])
plt.savefig("RNN_Multiset.png")