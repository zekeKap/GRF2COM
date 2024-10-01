import MLDataSequencing
import MLDataSequencingR2
import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#I need to use the MLDataSequencing to organize data from another dataset then feed it through the lastest model to see
#how much of an error they have. I think I am going to use magnitude to verify the error instead of individual x,y,z
x,y,trial_y=MLDataSequencingR2.Organizing_Data('C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/WBDS17walkT07.mat')
model=keras.models.load_model('C:/Users/kapla/Desktop/ZEKE NUEMove/Lateral Machine Learning/RNN_Model_2.keras')
#normalize the input first !
test=np.zeros((x.shape[1],x.shape[2]))
mocap=np.zeros((y.shape[1],y.shape[2]))
capturePredictedValues=np.ndarray([1000,2,3])
groundtruth=np.ndarray([1000,2,3])
print(test)
for i in range(0,1000):
    test=x[i,:,:]
    test=test.reshape((1,x.shape[1],x.shape[2]))
    #print(test)
    predicted_value=model.predict(test,verbose=1)
    capturePredictedValues[i,:,:]=predicted_value
    groundtruth[i,:,:]=y[i,:,:]
    print(predicted_value)
    # print(trial_y[i,:,:])
fig = plt.figure()
# syntax for 3-D projection
ax = plt.axes(projection='3d')
cogx=capturePredictedValues[:,1,0]
cogy=capturePredictedValues[:,1,1]
cogz=capturePredictedValues[:,1,2]
gtcogx=groundtruth[:,1,0]
gtcogy=groundtruth[:,1,1]
gtcogz=groundtruth[:,1,2]
# plotting
#ax.plot(cogx,cogy,cogz,color= 'blue')
ax.plot(gtcogx,gtcogy,gtcogz,color='green')
# ax.set_title('3D line plot geeks for geeks')
plt.show()
#here is where you should test the model predictions
