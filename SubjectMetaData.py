import pandas as pd
import numpy as np
import pickle
CSVRaw=pd.read_csv('C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/WBDSinfo.csv',header=0)
print(CSVRaw.columns)
SubjectMax=CSVRaw['Subject'].max()
SubjectMin=CSVRaw['Subject'].min()
Subject=[]
SubjectMass=[]
SubjectHeight=[]
for h in range(SubjectMin,SubjectMax):
    for i in range(0, len(CSVRaw['Subject'])):
        if (h==CSVRaw['Subject'][i]):
            Subject.append(h)
            SubjectMass.append(float(CSVRaw['Mass'][i]))
            SubjectHeight.append(float(CSVRaw['LegLength'][i]))
            break
print(Subject)
print(SubjectMass)
print(SubjectHeight)
SubjectMetaData=np.vstack([Subject,SubjectMass,SubjectHeight])
SubjectData=pd.DataFrame(np.transpose(SubjectMetaData),columns=['Subjects','Mass','Height'])
print(SubjectData)
SubjectData.to_csv("C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/SubjectMetaDataLegLength.csv")