import numpy as np
import MLDataSequencingR3
from matplotlib import pyplot as plt
import pickle
def COP_adjustment(data):
    COPx1 = np.array(data[:,  4])
    COPx2 = np.array(data[:,  9])
    COPz1 = np.array(data[:,  5])
    COPz2 = np.array(data[:,  10])
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
    x=np.array(data[:,:])
    x[:,  4]= COPx1
    x[:,  9]=COPx2
    x[:,  5]=COPz1
    x[:,  10]=COPz2
    return x

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
    Time=np.array(x[:,0])
    # Time2 = x[1:, :, 0]
    # Time1 = x[0:x.shape[0] - 1, :, 0]
    # delT = np.subtract(Time2, Time1)
    # Time = np.insert(delT, obj=2, values=np.mean(delT))
    # VarTime = np.ndarray(shape=(x.shape[0], 1), buffer=Time)
    # delT=np.full((x.shape[0]),float(0.01))
    # x[:, 0] = delT
    variable_names = ['Time', 'Fx1', 'Fy1', 'Fz1', 'COPx1', 'COPz1', 'Fx2', 'Fy2', 'Fz2', 'COPx2', 'COPz2', 'Height', 'Mass']
    MaxInput=np.array([0.02,2500,5000,2500,3.049,1.2227,2500,5000,2500,3.049,1.2227,1,100],dtype=np.float64)
    MinInput=np.array([0.001,-2500,-5000,-2500,1.249,0.7337,-2500,-5000,-2500,1.249,0.7337,0.5,40],dtype=np.float64)
    for j in range(len(variable_names)):
        scaled=(x[:,j]-MinInput[j])/(MaxInput[j]-MinInput[j])
        mean=np.mean(scaled)
        scaledMean = (mean - MinInput[j]) / (MaxInput[j] - MinInput[j])
        x[:,j]=(scaled*2)-1
    outputVar_names = ['COG_X', 'COG_Y', 'COG_Z']
    MaxOutput=np.array([3.049,2.5/2,1.227],dtype=np.float64)
    MinOutput=np.array([1.249,0.5,0.7337],dtype=np.float64)
    for k in range(len(outputVar_names)):
        scaled=(y[:,k]-MinOutput[k])/(MaxOutput[k]-MinOutput[k])
        mean=np.mean(scaled)
        scaledMean=(mean-MinOutput[k])/(MaxOutput[k]-MinOutput[k])
        y[:,k]=(scaled*2)-1
    x[:,0]=Time
    return x,y
def reshape(data,Timestep,Overlap):
    new_Timesteps=Timestep-Overlap
    depth=(data.shape[0]//new_Timesteps)-1
    reshaped_data = np.zeros((depth,Timestep, data.shape[1]))  # Shape: (n_new, 5, 15)
    for i in range(depth):
        reshaped_data[i,:,:]=data[i*Overlap:(i*Overlap)+Timestep,:]
    return reshaped_data

def reshapeinput(data,steps):
    n = data.shape[0]  # Number of original samples
    # Define time window parameters
    time_steps = steps  # 4 past + 1 new
    n_new = n - time_steps +1  # Number of valid sequences
    # Initialize new dataset
    reshaped_data = np.zeros((n_new, time_steps, data.shape[1]))  # Shape: (n_new, 5, 15)
    # Fill in the new dataset
    for i in range(n_new):
        reshaped_data[i] = np.vstack(data[i:i + time_steps])  # Stack 4 past + 1 new timestep
    #print("Final shape:", reshaped_data.shape)  # Expected: (n_new, 5, 15)
    return reshaped_data
