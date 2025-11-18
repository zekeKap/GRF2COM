import os, json
import numpy as np
import pandas as pd
import keras
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, TimeDistributed
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Your pipeline modules
import MLDataSequencingR4
import Data_AdjustmentMK2

#File paths
DATA_DIR = r"C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/Subject Based Walking 2"
META_CSV = r"C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/SubjectMetaDataLegLength.csv"

# Subject Selection
Train_Subjects=[1,2,3]
Val_Subjects=[11]
Test_Subjects=[14]
# HyerParameters
Timesteps=100
Overlap=25
Features=12
Output_Dimensions=3
Batch=32
Epochs=350
Learning_Rate=3e-4
units=256

Meta_Data = pd.read_csv(META_CSV)
def Subject_Coordination(Subjects):
    Input_Data = []
    Output_Data = []
    for Subject in range(0, len(Subjects)):
        File = DATA_DIR + "/Subject" + str(Subjects[Subject])
        print(Meta_Data["Subjects"][Subjects[Subject] - 1])
        Mass = Meta_Data["Mass"][Subjects[Subject] - 1]*9.81
        LegLength = Meta_Data["Height"][Subjects[Subject] - 1]
        Mat_Files = os.listdir(File)
        for Data_File in range(0, len(Mat_Files)):
            Trial_Data_File = File + "/" + Mat_Files[Data_File]
            InputTrain, OutputTrain = MLDataSequencingR4.Organize_Data(Trial_Data_File, LegLength, Mass)
            ProcessedInput, ProcessedOutput=PreProcessingData(InputTrain,OutputTrain)
            Input_Data.append(ProcessedInput)
            Output_Data.append(ProcessedOutput)
    X=np.concatenate(Input_Data,axis=0)
    Y=np.concatenate(Output_Data,axis=0)
    X=np.delete(X,0,axis=2)
    return X, Y
def PreProcessingData(X_Raw,Y_Raw):
    # x_adj = Data_Adjustment.COP_adjustment(X_Raw)
    # Xn, Yn = Data_Adjustment.Data_NormalizationMK3(x_adj, Y_Raw)  # normalized inputs & labels
    # # Window inputs & targets
    # x_win = Data_Adjustment.reshapeinput(Xn, Timesteps)  # (N, T, FEATS) in your pipeline
    # y_win = Data_Adjustment.reshapeinput(Yn, Timesteps)  # (N, 1, 3) aligned to last frame Yn is Normalized Trying Raw data here
    # #y_win = y_win.reshape(y_win.shape[0], y_win.shape[-1])  # (N, 3)
    # return x_win,y_win
    x_Cop=Data_AdjustmentMK2.COP_adjustment(X_Raw)
    x_Norm, y_Norm=Data_AdjustmentMK2.Data_NormalizationMK3(x_Cop,Y_Raw)
    x_win=Data_AdjustmentMK2.reshape(x_Norm,Timesteps,Overlap)
    y_win=Data_AdjustmentMK2.reshape(y_Norm,Timesteps,Overlap)
    return x_win, y_win

# Train_Input_Data_List, Train_Output_Data_List=Subject_Coordination(Train_Subjects)
# Val_Input_Data_List, Val_Output_Data_List=Subject_Coordination(Val_Subjects)
# Test_Input_Data_List, Test_Output_Data_List=Subject_Coordination(Test_Subjects)
# Train_Input_Data_Array=np.concatenate((Train_Input_Data_List))
# Train_Output_Data_Array=np.concatenate((Train_Output_Data_List))
# Val_Input_Data_Array=np.concatenate((Val_Input_Data_List))
# Val_Output_Data_Array=np.concatenate((Val_Output_Data_List))
# Test_Input_Data_Array=np.concatenate((Test_Input_Data_List))
# Test_Output_Data_Array=np.concatenate((Test_Output_Data_List))
# print(Train_Input_Data_Array.shape)
# X_Train, Y_Train=PreProcessingData(Train_Input_Data_Array,Train_Output_Data_Array)
# X_Val, Y_Val= PreProcessingData(Val_Input_Data_Array,Val_Output_Data_Array)
# X_Test, Y_Test=PreProcessingData(Test_Input_Data_Array,Test_Output_Data_Array)
X_Train, Y_Train=Subject_Coordination(Train_Subjects)
X_Val, Y_Val=Subject_Coordination(Val_Subjects)
X_Test, Y_Test=Subject_Coordination(Test_Subjects)
print(X_Train.shape)
#Model Building
model = Sequential([
        keras.Input((Timesteps, Features),batch_size=Batch, dtype= "float32"),
        LSTM(units, activation='tanh',recurrent_activation='sigmoid', return_sequences=True, recurrent_dropout=0.05, dropout=0.15),
        # keras.layers.Dropout(0.1),
        keras.layers.LayerNormalization(),
        # Dense((units*2), activation='relu'),
        # keras.layers.LayerNormalization(),
        TimeDistributed(Dense(units, activation='linear')),
        # keras.layers.LayerNormalization(),
        TimeDistributed(Dense(Output_Dimensions, activation='linear'))
    ])
opt = keras.optimizers.Adam(learning_rate=Learning_Rate)
model.compile(optimizer=opt, loss='mse', metrics=['root_mean_squared_error'])
model.summary()

early_stop = EarlyStopping(
    monitor='val_loss', mode='min',
    patience=50, min_delta=5e-5,
    restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', mode='min', min_delta=5e-5,
    factor=0.8, patience=5,
    min_lr=1e-6, verbose=1
)
ckpt = ModelCheckpoint(
    'best_model_dr0.1.keras', monitor='val_loss', mode='min', save_best_only=True, verbose=1
)
history = model.fit(
    X_Train, Y_Train,
    validation_data=(X_Val, Y_Val),
    epochs=Epochs,
    batch_size=Batch,
    shuffle=True,                     # shuffle samples (windows), not timesteps
    callbacks=[early_stop, reduce_lr,ckpt],
    verbose=2
)

# --------------------
# Quick verification plots
# --------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# --------------------
# Final held-out test (optional)
# --------------------

test_mse, test_rmse = model.evaluate(X_Test, Y_Test, verbose=0)
print(f"\nHELD-OUT TEST — MSE: {test_mse:.6f}  RMSE: {test_rmse:.6f}")
test_data=model.predict(X_Test, verbose=0)
plt.plot(test_data[:,-1,0], label="Pred Com-X")
plt.plot(Y_Test[:,-1, 0], label='True CoM-X')
plt.legend()
plt.show()
