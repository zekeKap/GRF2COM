import tensorflow as tf
import numpy as np
import MLDataSequencingR3
from matplotlib import pyplot as plt
import pickle
def COP_adjustment(data):
    COPx1 = np.array(data[:, 0, 4])
    COPx2 = np.array(data[:, 0, 9])
    COPz1 = np.array(data[:, 0, 5])
    COPz2 = np.array(data[:, 0, 10])
    for i in range(0, len(COPx1) - 3):
        deltaCOP = float(abs(COPx1[i + 1] - COPx1[i])) / float(COPx1[i])
        if (deltaCOP > float(0.1)):
            COPx1[i + 1] = (COPx1[i] + COPx1[i + 2]) / 2
            # print(deltaCOP)
    for i in range(0, len(COPx2) - 3):
        deltaCOP = float(abs(COPx2[i + 1] - COPx2[i])) / float(COPx2[i])
        if (deltaCOP > float(0.1)):
            COPx2[i + 1] = (COPx2[i] + COPx2[i + 2]) / 2
            # print(deltaCOP)
    for i in range(0, len(COPz1) - 3):
        deltaCOP = float(abs(COPz1[i + 1] - COPz1[i])) / float(COPz1[i])
        if (deltaCOP > float(0.1)):
            COPz1[i + 1] = (COPz1[i] + COPz1[i + 2]) / 2
            # print(deltaCOP)
    for i in range(0, len(COPz2) - 3):
        deltaCOP = float(abs(COPz2[i + 1] - COPz2[i])) / float(COPz2[i])
        if (deltaCOP > float(0.1)):
            COPz2[i + 1] = (COPz2[i] + COPz2[i + 2]) / 2
            # print(deltaCOP)
    x=np.array(data[:,:,:])
    x[:, 0, 4]= COPx1
    x[:, 0, 9]=COPx2
    x[:, 0, 5]=COPz1
    x[:, 0, 10]=COPz2
    return x
def Data_Normalization(DataX,DataY):
    x=np.array(DataX)
    y=np.array(DataY)
    Data = ['', '', '', '']
    file_type = '.pickle'
    Save_file_Names = ['MaxInput', 'MinInput', 'MaxOutput', 'MinOutput']
    variable_names = ['Time','Fx1', 'Fy1', 'Fz1', 'COPx1', 'COPz1', 'Fx2', 'Fy2', 'Fz2', 'COPx2', 'COPz2','Height','Mass']
    outputVar_names = ['COG_X', 'COG_Y', 'COG_Z']
    for i in range(4):
        with open(Save_file_Names[i] + file_type, 'rb') as file:
            Data[i] = pickle.load(file)
    for j in range(len(variable_names)):
        string = variable_names[j]
        x[:, :, j] = (x[:, :, j] - Data[1]['Min' + string]) / (Data[0]['Max' + string] - Data[1]['Min' + string])
        x[:,:,j]=x[:,:,j]*(1-(-1))+(-1)
    for k in range(len(outputVar_names)):
        string = outputVar_names[k]
        y[:, :, k] = (y[:, :, k] - Data[3]['outMin' + string]) / (Data[2]['outMax' + string] - Data[3]['outMin' + string])
        y[:,:,k]=y[:,:,k]*(1-(-1))+(-1)
    return x, y

import numpy as np
import pickle


def Data_NormalizationMK2(DataX, DataY):
    x = np.array(DataX)
    y = np.array(DataY)

    # Load saved min/max data from pickle files
    file_type = '.pickle'
    Save_file_Names = ['MaxInput', 'MinInput', 'MaxOutput', 'MinOutput']
    Data = ['', '', '', '']
    for i in range(4):
        with open(Save_file_Names[i] + file_type, 'rb') as file:
            Data[i] = pickle.load(file)

    # Variable names
    variable_names = ['Time', 'Fx1', 'Fy1', 'Fz1', 'COPx1', 'COPz1', 'Fx2', 'Fy2', 'Fz2', 'COPx2', 'COPz2', 'Height',
                      'Mass']
    outputVar_names = ['COG_X', 'COG_Y', 'COG_Z']

    # Normalize input variables
    for j in range(len(variable_names)):
        string = variable_names[j]
        # Step 1: Min-Max scaling
        scaled = (x[:, :, j] - Data[1]['Min' + string]) / (Data[0]['Max' + string] - Data[1]['Min' + string])
        # Step 2: Center and rescale
        mean_scaled = np.mean(scaled)
        x[:, :, j] = (scaled - mean_scaled) * 2

    # Normalize output variables
    for k in range(len(outputVar_names)):
        string = outputVar_names[k]
        # Step 1: Min-Max scaling
        scaled = (y[:, :, k] - Data[3]['outMin' + string]) / (Data[2]['outMax' + string] - Data[3]['outMin' + string])
        # Step 2: Center and rescale
        mean_scaled = np.mean(scaled)
        y[:, :, k] = (scaled - mean_scaled) * 2

    return x, y
def Data_NormalizationMK3(DataX,DataY):
    ForceXZmax=2500
    ForceYmax=5000
    TreadmillminX=1.249
    TreadmillminY=0
    TreadmillminZ=0.7337
    TreadmillmaxX=3.049
    TreadmillmaxY=0
    TreadmillmaxZ=1.2227
    HeightMin=1
    HeightMax=2.5
    MassMin=40
    MassMax=100
    x = np.array(DataX)
    y = np.array(DataY)
    Time2 = x[1:, :, 0]
    Time1 = x[0:x.shape[0] - 1, :, 0]
    delT = np.subtract(Time2, Time1)
    Time = np.insert(delT, obj=2, values=np.mean(delT))
    VarTime = np.ndarray(shape=(x.shape[0], 1), buffer=Time)
    x[:, :, 0] = VarTime
    variable_names = ['Time', 'Fx1', 'Fy1', 'Fz1', 'COPx1', 'COPz1', 'Fx2', 'Fy2', 'Fz2', 'COPx2', 'COPz2', 'Height', 'Mass']
    MaxInput=np.array([0,2500,5000,2500,3.049,1.2227,2500,5000,2500,3.049,1.2227,2.5,100],dtype=np.float64)
    MinInput=np.array([0,0,0,0,1.249,0.7337,0,0,0,1.249,0.7337,1,40],dtype=np.float64)
    for j in range(1,len(variable_names)):
        scaled=(x[:,:,j]-MinInput[j])/(MaxInput[j]-MinInput[j])
        mean=(MinInput[j]+MaxInput[j])/2
        x[:,:,j]=(scaled*2)-1
    outputVar_names = ['COG_X', 'COG_Y', 'COG_Z']
    MaxOutput=np.array([3.049,2.5/2,1.227],dtype=np.float64)
    MinOutput=np.array([1.249,0.5,0.7337],dtype=np.float64)
    for k in range(len(outputVar_names)):
        scaled=(y[:,:,k]-MinOutput[k])/(MaxOutput[k]-MinOutput[k])
        mean=(MinOutput[k]+MaxOutput[k])/2
        y[:,:,k]=(scaled*2)-1
    return x,y

def reshape(data):
    n = data.shape[0]  # Number of original samples
    # Define time window parameters
    time_steps = 5  # 4 past + 1 new
    n_new = n - 4  # Number of valid sequences
    # Initialize new dataset
    reshaped_data = np.zeros((n_new, time_steps, data.shape[2]))  # Shape: (n_new, 5, 15)
    # Fill in the new dataset
    for i in range(n_new):
        reshaped_data[i] = np.vstack(data[i:i + time_steps])  # Stack 4 past + 1 new timestep
    #print("Final shape:", reshaped_data.shape)  # Expected: (n_new, 5, 15)
    return reshaped_data
def reshapeoutput(data,steps):
    n=data.shape[0]
    n_new=n-(steps-1)
    reshaped_data=np.zeros((n_new,data.shape[1],data.shape[2]))
    for i in range(n_new):
        reshaped_data[i,:,:]=data[i+(steps-1),:,:]
    return reshaped_data