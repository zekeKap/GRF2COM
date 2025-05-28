# We gotta make this into a function to call
import numpy as np


def MKRPairing(df):
    import pandas as pd
    result_list = []
    # Iterate through the DataFrame in steps of 3
    for i in range(0, len(df), 3):
        # Combine the first two rows by summing them
        combined_row = ((df.iloc[i+1] - df.iloc[i])/float((df.iat[i+1,0]-df.iat[i,0])))*float(((df.iat[i,0]+df.iat[i+1,0])/2)-df.iat[i,0])+df.iloc[i]
        # Append the combined row to the result list
        result_list.append(combined_row)
        # Append the third row unchanged to the result list
        result_list.append(df.iloc[i + 2])
    # Convert the result list to a DataFrame using pd.concat
    result_df = pd.concat(result_list, axis=1).T.reset_index(drop=True)
    return(result_df)
def output_single_shape(output,samples):
    import pandas as pd
    output=pd.DataFrame.to_numpy(output)
    y=np.ndarray([samples-1,1,3])
    for i in range(1,samples):
        y[i-1,:,:]=output[i,:]
    # print(y)
    return y
def input_single_shape(input,samples):
    x=np.ndarray([samples-1,1,input.shape[1]])
    for i in range(1,samples):
        x[i-1,:,:] = input[i,:]
    # print(x)
    return x
def Input_combine_output(Input,output,sample):
    input = np.zeros([sample - 1, 1, 18])
    # print(input.shape)
    for j in range(0,sample-1):
        input[j,:,0:15]=Input[j,:,:]
    for i in range(1,sample-1):
        input[i,0,15]=output[i-1,0,0]
        input[i,0,16]=output[i-1,0,1]
        input[i,0,17]=output[i-1,0,2]
    return input
def Organizing_Data(file,Height,Mass):
    import scipy
    import pandas as pd
    import numpy as np
    import sys
    import time
    Data = scipy.io.loadmat(file)
    GRF = Data['Dataset'][0, 0]  # importing the GRF data from Mat file. THis is gives us the Ground reaction forces
    MKR = Data['Dataset'][0, 1]  # importing the MKR(Marker Data) from the Mat file. This data gives us the COG data
    GRFdf = pd.DataFrame(GRF)
    GRFcolumnname = ['Time', 'Fx1', 'Fy1', 'Fz1', 'COPx1', 'COPy1', 'COPz1', 'Ty1',
                     'Fx2', 'Fy2', 'Fz2', 'COPx2', 'COPy2', 'COPz2', 'Ty2']  # Column Titles for the GRF dataframe
    GRFdf.columns = GRFcolumnname  # Titling the GRF Columns
    GRFdf["Time"]=GRFdf["Time"]/100
    MKRdf = pd.DataFrame(MKR)
    MKRcolumnname = ['Time','R.ASISX','R.ASISY','R.ASISZ','L.ASISX','L.ASISY','L.ASISZ',
   	'R.PSISX','R.PSISY','R.PSISZ','L.PSISX','L.PSISY','L.PSISZ','L.Iliac.CrestX',
    'L.Iliac.CrestY','L.Iliac.CrestZ','R.Iliac.CrestX',	'R.Iliac.CrestY','R.Iliac.CrestZ',
     'R.GTRX','R.GTRY','R.GTRZ','R.KneeX','R.KneeY','R.KneeZ','R.HFX','R.HFY','R.HFZ','R.TTX',
 	'R.TTY','R.TTZ','R.AnkleX',	'R.AnkleY',	'R.AnkleZ',	'R.HeelX','R.HeelY',
    'R.HeelZ','R.MT1X',	'R.MT1Y','R.MT1Z','R.MT5X','R.MT5Y','R.MT5Z','L.GTRX',
    'L.GTRY','L.GTRZ','L.KneeX','L.KneeY','L.KneeZ','L.HFX','L.HFY','L.HFZ',
    'L.TTX','L.TTY','L.TTZ','L.AnkleX','L.AnkleY','L.AnkleZ','L.HeelX',	'L.HeelY',
    'L.HeelZ','L.MT1X','L.MT1Y','L.MT1Z','L.MT5X','L.MT5Y','L.MT5Z']  # Column Titles for the MKR dataframe
    MKRdf.columns = MKRcolumnname  # Titling the MKR dataframe

    # we insert the COG calculations here
    def COGX(RASISX, LASISX, RPSISX, LPSISX):
        return (RASISX +LASISX+RPSISX+LPSISX) / 4000

    def COGY(RASISY, LASISY, RPSISY, LPSISY):
        return (RASISY+LASISY+RPSISY+LPSISY)/ 4000

    def COGZ(RASISZ, LASISZ, RPSISZ, LPSISZ):
        return (RASISZ+LASISZ+RPSISZ+LPSISZ) / 4000

    MKRdf['COG_X'] = MKRdf.apply(lambda column: COGX(column['R.ASISX'], column['L.ASISX'], column['R.PSISX'], column['L.PSISX']),
                                 axis=1)
    MKRdf['COG_Y'] = MKRdf.apply(lambda column: COGY(column['R.ASISY'], column['L.ASISY'], column['R.PSISY'], column['L.PSISY']),
                                 axis=1)
    MKRdf['COG_Z'] = MKRdf.apply(lambda column: COGZ(column['R.ASISZ'], column['L.ASISZ'], column['R.PSISZ'], column['L.PSISZ']),
                                 axis=1)

    MKRdf=MKRPairing(MKRdf)
    # print("MKRdf:")
    # print(MKRdf)
    # wait=input("Press Enter to Continue")
    #The last 30s of the trial is what is being recorded
    if len(MKRdf)<len(GRFdf):
        GRFdf=GRFdf.truncate(before=(len(GRFdf)-len(MKRdf)))
    # MKRdf.to_csv(path_or_buf="MKRdf.csv")
    # GRFdf.to_csv(path_or_buf="GRFdf.csv")
    def Create_Sequence(data, columnSeperation):
        # x=np.zeros(shape=(2,11),dtype=float,order='C')
        # y=np.zeros(shape=(2,3),dtype=float,order='C')
        x = np.vstack((data[0, 0:columnSeperation], data[1, 0:columnSeperation]))  # first 2 rows of input training data
        y = np.vstack((data[0, columnSeperation:], data[1, columnSeperation:]))  # first 2 rows of output training data
        for i in range(2, len(data)):
            int_X = np.vstack((data[i - 1, 0:columnSeperation], data[i,0:columnSeperation]))  # sequencing data for input the 2 columns being made are a copy of previous and the next one in data
            int_Y = np.vstack((data[i - 1, columnSeperation:], data[i,columnSeperation:]))  # sequencing data for output the 2 columns being made are a copy of previous and the next one in data
            x = np.vstack((x, int_X))  # Appending the columns
            y = np.vstack((y, int_Y))  # Appending the columns
        return x, y

    def Reshape(x, y,samples):  # reshape traing data so that the 2D array is 2 columns and the depth is equal to the number of data points - 1
        count = 0
        final_X = np.zeros(shape=(samples - 1, 2, x.shape[1]))
        final_Y = np.zeros(shape=(samples - 1, 2, y.shape[1]))
        # print("Final_y")
        # print(final_Y)
        for i in range(1, len(x), 2):
            # print("Reshape Print")
            X = np.vstack((x[i - 1, :], x[i, :]))
            Y = np.vstack((y[i - 1, :], y[i, :]))
            final_X[count] = X
            final_Y[count] = Y
            count = count + 1
            # print(count)
        return final_X, final_Y
    # Input_Data = GRFdf[['Fx1', 'Fy1', 'Fz1', 'COPx1', 'COPz1', 'Ty1', 'Fx2',
    #                     'Fy2', 'Fz2', 'COPx2', 'COPz2', 'Ty2']]  # Input data dataframe
    Input_Data = GRFdf[['Time','Fx1', 'Fy1', 'Fz1','COPx1','COPz1','Fx2',
                        'Fy2', 'Fz2','COPx2','COPz2']]  # Input data dataframe

    Input_Data.loc[:,'COPx1'] = Input_Data['COPx1']/1000
    # # Input_Data['COPy1']=Input_Data['COPy1']/1000
    Input_Data.loc[:,'COPz1'] = Input_Data['COPz1']/1000
    Input_Data.loc[:,'COPx2'] = Input_Data['COPx2']/1000
    # # Input_Data['COPy2']=Input_Data['COPy2']/1000
    Input_Data.loc[:,'COPz2'] = Input_Data['COPz2']/1000
    # height_array=np.full((len(Input_Data),1), float(input("Height of Subject :")))
    # mass_array=np.full((len(Input_Data),1),  float(input("Weight of Subject: ")))
    height_array = np.full((len(Input_Data), 1), float(Height))
    mass_array = np.full((len(Input_Data), 1), float(Mass))
    Input_Data.insert(loc=Input_Data.shape[1], column="Height",value=height_array)
    Input_Data.insert(loc=Input_Data.shape[1], column="Mass", value=mass_array)
    Input_DataR3 = pd.DataFrame.to_numpy(Input_Data)  # placeholder for trial
    # print("Input_Data")
    # print(Input_Data)
    # wait=input("Press Enter to Continue")
    Output_Data = MKRdf[['COG_X', 'COG_Y', 'COG_Z']]  # Output data dataframe
    # print("Output_Data")
    # print(Output_Data)
    # wait=input("Press Enter to Continue")
    Input_Data.insert(loc=Input_Data.shape[1], column='COG_X', value=Output_Data['COG_X'].to_numpy(dtype=float))  # combine input and output dataframe for sequencing
    Input_Data.insert(loc=Input_Data.shape[1], column='COG_Y', value=Output_Data['COG_Y'].to_numpy(dtype=float))  # combine input and output dataframe for sequencing
    Input_Data.insert(loc=Input_Data.shape[1], column='COG_Z', value=Output_Data['COG_Z'].to_numpy(dtype=float))  # combine input and output dataframe for sequencing
    # print("Input Data Final :")
    # print(Input_Data)
    # wait=input("Press Enter to Continue")
    Input_Data_Array=Input_Data.to_numpy(dtype=np.float64)
    x, y = Create_Sequence(Input_Data_Array, int(16))
    np.printoptions(precision=3, suppress=True)
    # print(x.shape)
    # print(x)
    # print("Y :")
    # print(y.shape)
    # print(y)
    # time.sleep(60)
    # x_reshape, y_reshape = Reshape(x, y, Input_Data.shape[0])
    y_reshape=output_single_shape(Output_Data,Input_Data.shape[0])
    x_reshape=input_single_shape(Input_DataR3,3000)
    # x_reshape=Input_combine_output(x_reshape,y_reshape,3000)
    # np.set_printoptions(precision=3, threshold=sys.maxsize, suppress=True)
    # # print(x_reshape)
    # print(y_reshape)
    x_reshape.astype(np.float64)
    y_reshape.astype(np.float64)
    nan_coordinates= np.isnan(y_reshape)
    nan_z=np.empty((nan_coordinates.shape[0]),dtype=int)
    counter=0
    for k in range(0,int(nan_coordinates.shape[0])):
        if nan_coordinates[k,0,0]:
            nan_z[counter]=int(k)
            counter=counter+1
    nanz=nan_z[:counter]
    # print(nanz)
    y_reshape=np.delete(y_reshape,nanz,axis=0)
    x_reshape=np.delete(x_reshape,nanz,axis=0)
    # print(x_reshape.shape)
    # print(y_reshape.shape)
    trial_Y=Output_Data.to_numpy(dtype=np.float64).reshape(3000,1,3)
    trial_Y=trial_Y[:-1,:,:]
    return x_reshape, y_reshape,trial_Y
# file='C:/Users/kapla/Desktop/ZEKE NUEMove/MachineLearningDataSet/WBDS01walkT01.mat'
# x_reshape, y_reshape, Input_Data_Array, Input_Data=Organizing_Data(file)