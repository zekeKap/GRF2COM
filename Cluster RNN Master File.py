# I need to change Lateral Machine Learning into a function and Call it with all the arguments necessary
# everything else is a function basically
# I need to make a directory of all the usable .mat files
#
import os
import pandas as pd
import MLDataSequencingR3
import keras
from keras.api.layers import SimpleRNN, Dense
import keras.api.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler 
def absError(y_true,y_pred):
 f_y_pred=keras.ops.numpy.ravel(y_pred)
 f_y_true=keras.ops.numpy.ravel(y_true)
 e=tf.math.reduce_sum(f_y_pred-f_y_true)
 return abs(e)
unit= int(128)
DataSet=os.listdir("/scratch/kaplan.ez/WalkingDataSet")
Path_Dataset= "/scratch/kaplan.ez/WalkingDataSet/"
DatasetInfoFile="WBDSinfo.csv"
DatasetInfo=pd.read_csv(DatasetInfoFile)
print(DatasetInfo)
Subject=[]
for i in range(len(DataSet)):
    subject=DataSet[i][4:6]
    Subject.append(int(subject))
print(Subject)
input = keras.Input(shape=(1, 15), batch_size=1000)

#x = SimpleRNN(1, activation='linear', return_sequences=True)(x)  # Making this Sigmoind took out the shoot out
#x = SimpleRNN(1, activation='linear', return_sequences=True)(x)  # This was tanh before tanh and next iteration is tanh.tanh increase complexit
#x = SimpleRNN(1, activation='linear', return_sequences=True)(x)
#x = SimpleRNN(1, activation='linear', return_sequences=True)(x)
#x = SimpleRNN(1, activation='linear', return_sequences=True)(x)
#x = SimpleRNN(1, activation='linear', return_sequences=True)(x)
#x = SimpleRNN(1, activation='linear', return_sequences=True)(x)
#x = SimpleRNN(1, activation='linear', return_sequences=True)(x)
#x = SimpleRNN(1, activation='linear', return_sequences=True)(x)
#x = SimpleRNN(unit, activation='tanh', return_sequences=True)(x)
#x = SimpleRNN(unit, activation='tanh', return_sequences=True)(x)  # This was tanh before tanh and next iteration is tanh.tanh increase complexit
#x = SimpleRNN(unit, activation='tanh', return_sequences=True)(x)
#x = SimpleRNN(unit, activation='tanh', return_sequences=True)(x)
#x = SimpleRNN(unit, activation='tanh', return_sequences=True)(x)
#x = SimpleRNN(unit, activation='tanh', return_sequences=True)(x)
#x = SimpleRNN(unit, activation='tanh', return_sequences=True)(x)
x = SimpleRNN(unit, activation='tanh', return_sequences=True)(input)
x=keras.layers.Dropout(0.2)(x)
x = SimpleRNN(unit, activation='tanh', return_sequences=True)(x)
x = keras.layers.Dropout(0.2)(x)
x = SimpleRNN(unit, activation='tanh', return_sequences=False)(x)
x=keras.layers.Dropout(0.2)(x)
output=Dense(units=1,activation='linear')(x)
K.clear_session()
model=keras.Model(inputs=input,outputs=output)
model.compile(optimizer= 'adam',loss=keras.losses.MeanSquaredError('mean'),metrics=[]) #RMSE solves our scaling problem
model.summary()
# I now need to have the subject dictate the automated part of the Height and Weight parts
# the question is how should I set this up I can do the hard code with the for loop
for z in range(20):
 np.random.shuffle(DataSet)
 for k in range(len(DataSet)):
     for j in range(len(DatasetInfo)-1):
         if Subject[k]== DatasetInfo["Subject"][j]:
             Height=float(DatasetInfo["Height"][j]/100)
             Mass=float(DatasetInfo["Mass"][j])
             file=Path_Dataset+DataSet[k]
             x_reshape, y_reshape, trial_y = MLDataSequencingR3.Organizing_Data(file,float(Height),float(Mass))#Add Height and Mass
             x_reshape=x_reshape.astype(np.float64)
             y_reshape=y_reshape.astype(np.float64)
             print(file)
             break
     percent=float((k/len(DataSet))*100)
     scalerX=MinMaxScaler(feature_range=(0,1))
     x=x_reshape.reshape(-1,x_reshape.shape[-1])
     x=scalerX.fit_transform(x)
     x=x.reshape(x_reshape.shape)
     scalerY=MinMaxScaler(feature_range=(0,1))
     y=y_reshape.reshape(-1,y_reshape.shape[-1])
     y=scalerY.fit_transform(y)
     y=y.reshape(y_reshape.shape)
     y=y[:,0,0].reshape(y_reshape.shape[0],1,1)
     model.reset_metrics()
     model.fit(x, y, batch_size=25, epochs=1, validation_split=0.25,shuffle=False)  # larger batch size the better it is
     print(percent)
    
model.save('RNN_Model_Multi_Set.keras')
capturePredictedValues=model.predict(x,batch_size=1)
# cogx=(capturePredictedValues*(y_max-y_min))+(y_min)
plt.plot(capturePredictedValues[:,0],'o')
plt.plot(y[:,0,0],'o')
# plt.plot(error,'o')
plt.legend(["predict","GT"])
plt.savefig("RNN_Multiset.png")