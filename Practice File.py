import Data_Adjustment
import Data_AdjustmentMK2
import MLDataSequencingR3
import MLDataSequencingR4
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os
DATA_DIR = r"C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/Subject Based Walking 2"
META_CSV = r"C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/SubjectMetaDataLegLength.csv"

# Subject Selection
Train_Subjects=[4]
Val_Subjects=[]
Test_Subjects=[]
Meta_Data = pd.read_csv(META_CSV)
Timesteps=100
Overlap=50
Features=12
Output_Dimensions=3
Batch=100
Epochs=350
Learning_Rate=3e-4
units=128
def Subject_Coordination(Subjects):
    Input_Data = []
    Output_Data = []
    for Subject in range(0, len(Subjects)):
        File = DATA_DIR + "/Subject" + str(Subjects[Subject])
        print(Meta_Data["Subjects"][Subjects[Subject] - 1])
        Mass = Meta_Data["Mass"][Subjects[Subject] - 1]
        LegLength = Meta_Data["Height"][Subjects[Subject] - 1]
        Mat_Files = os.listdir(File)
        for Data_File in range(0, len(Mat_Files)):
            Trial_Data_File = File + "/" + Mat_Files[Data_File]
            print(Trial_Data_File)
            #InputTrain, OutputTrain, TrashY = MLDataSequencingR3.Organizing_Data(Trial_Data_File, LegLength, Mass)
            InputTrain, OutputTrain=MLDataSequencingR4.Organize_Data(Trial_Data_File,LegLength, Mass)
            ProcessedInput, ProcessedOutput=PreProcessingData(InputTrain,OutputTrain)
            Input_Data.append(ProcessedInput)
            Output_Data.append(ProcessedOutput)
    # for i in range(len(Input_Data)):
    #     print(Input_Data[i].shape)
    #     X=Input_Data[i]
    #     plt.plot(X[:,:,0],X[:,:,1], label="FX1")
    #     plt.legend()
    #     plt.title("FX1")
    #     plt.show()
    print(Input_Data[0].shape)
    X=np.concatenate(Input_Data,axis=0)
    Y=np.concatenate(Output_Data,axis=0)
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
X,Y=Subject_Coordination(Train_Subjects)
#X2,Y2=Subject_Coordination(Val_Subjects)
# print(X.shape)
# print(X[0,0,0])
# print(X[0,:,0])
print(np.isnan(X).any())
plt.plot(X[0,:,0],X[0,:,1], label="FX1", marker='o')
plt.plot(X[0,:,0],X[0,:,6], label="FX2", marker='x')
# # #plt.plot(Y2[:,0,0], label="Y2")
plt.legend()
# plt.title("FX1")
plt.show()
# plt.cla()
# plt.plot(Y[:,0,1], label="Y")
# #plt.plot(Y2[:,0,1], label="Y2")
# plt.legend()
# plt.title("COMY")
# plt.show()
# plt.cla()
# plt.plot(Y[:,0,2], label="Y")
# #plt.plot(Y2[:,0,2], label="Y2")
# plt.legend()
# plt.title("COMZ")
# plt.show()
