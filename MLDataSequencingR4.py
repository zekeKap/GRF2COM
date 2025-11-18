import pandas as pd
import numpy as np
import scipy
import heapq
import matplotlib.pyplot as plt
def COG_Linearization(COG,TimeMKR,TimeABS):
    COG_CALC=[]
    TimeMKR=np.array(TimeMKR)
    TimeABS=np.array(TimeABS)
    COG=np.array(COG)
    counter=0
    for i in range (len(TimeABS)):
        deltas=abs(TimeABS[i]-TimeMKR)
        zero_index=np.where(deltas==float(0))
        if (zero_index!=np.empty([])):
            COG_CALC.append(COG[int(zero_index[0])])
        else:
            s1=heapq.nsmallest(1,deltas)
            List_Index=np.where(deltas==s1)[0]
            Index_TMKRa=int(List_Index[0])
            if(len(List_Index)>1):
                Index_TMKRb=Index_TMKRa-1
            else:
                Index_TMKRb=Index_TMKRa-2
            TA =TimeMKR[Index_TMKRa]
            TB=TimeMKR[Index_TMKRb]
            T0=min(TA,TB)
            T1=max(TA,TB)
            COG0_Index=np.where(TimeMKR == T0)
            COG0=float(COG[COG0_Index])
            COG1_Index=np.where(TimeMKR == T1)
            COG1 =float(COG[COG1_Index])
            slope=(COG1-COG0)/(T1-T0)
            COG_Estimate=float((slope*(TimeABS[i]-T0))+COG0)
            COG_CALC.append(COG_Estimate)
    Output=np.transpose(np.array([COG_CALC]))
    return Output
def COG_LinearizationMK2(COG,TimeMKR,TimeABS,fill='edge'):
    """
    Resample COG_mat (T_MKR x F) onto TimeABS.
    Returns ndarray (len(TimeABS), F).
    """
    tA = np.asarray(TimeABS, dtype=np.float64).ravel()
    tM = np.asarray(TimeMKR, dtype=np.float64).ravel()
    Y  = np.asarray(COG, dtype=np.float64)

    # Normalize Y to 2D
    if Y.ndim == 1:
        Y = Y[:, None]

    # Try to align the time dimension
    if Y.shape[0] == tM.size:
        pass  # already (T_MKR, F)
    elif Y.shape[1] == tM.size:
        Y = Y.T  # was (F, T_MKR)
    else:
        raise ValueError(
            f"TimeMKR length {tM.size} doesn't match any axis of COG_mat {Y.shape}. "
            "Build COG from the same cleaned rows as MKRdf['Time'] or select the correct axis."
        )

    # Drop rows where time or any column is non-finite
    row_mask = np.isfinite(tM) & np.isfinite(Y).all(axis=1)
    tM, Y = tM[row_mask], Y[row_mask, :]

    if tM.size < 2:
        # Not enough points to interpolate
        return np.full((tA.size, Y.shape[1]), np.nan, np.float64)

    # Sort by time and deduplicate
    order = np.argsort(tM)
    tM, Y = tM[order], Y[order, :]
    keep = np.r_[True, np.diff(tM) > 0]
    tM, Y = tM[keep], Y[keep, :]

    # Interpolate each column
    out = np.empty((tA.size, Y.shape[1]), dtype=np.float64)
    for j in range(Y.shape[1]):
        if fill == 'nan':
            out[:, j] = np.interp(tA, tM, Y[:, j], left=np.nan, right=np.nan)
        else:
            out[:, j] = np.interp(tA, tM, Y[:, j])
    return out
def NaN_Linearization(NaNcoordinates,Time,Output):
    #print(NaNcoordinates)
    T=np.asarray(Time[np.logical_not(np.isnan(Output[:,0]))]).ravel()
    Clean_Data=Output[np.logical_not(np.isnan(Output[:,0]))]
    for i in range(len(NaNcoordinates)):
        Point=T[int(NaNcoordinates[i])]
        COG_X=np.asarray(Clean_Data[:,0]).ravel()
        COG_X_Point=np.interp(Point,T,COG_X)
        COG_Y = np.asarray(Clean_Data[:, 1]).ravel()
        COG_Y_Point = np.interp(Point, T, COG_Y)
        COG_Z = np.asarray(Clean_Data[:, 2]).ravel()
        COG_Z_Point = np.interp(Point, T, COG_Z)
        Output[NaNcoordinates[i]]=np.array([COG_X_Point,COG_Y_Point,COG_Z_Point])
        #print(COG_X_Point)
    #print(Output.shape)
    return Output
def Organize_Data(file,Hieght,Mass):
    Data = scipy.io.loadmat(file)
    GRF = Data['Dataset'][0, 0]  # importing the GRF data from Mat file. THis is gives us the Ground reaction forces
    MKR = Data['Dataset'][0, 1]  # importing the MKR(Marker Data) from the Mat file. This data gives us the COG data
    GRFdf = pd.DataFrame(GRF, dtype=float) #Converts the Mat to dataframe
    GRFcolumnname = ['Time', 'Fx1', 'Fy1', 'Fz1', 'COPx1', 'COPy1', 'COPz1', 'Ty1',
                     'Fx2', 'Fy2', 'Fz2', 'COPx2', 'COPy2', 'COPz2', 'Ty2']  # Column Titles for the GRF dataframe
    GRFdf.columns = GRFcolumnname  # Titling the GRF Columns
    MKRdf = pd.DataFrame(MKR, dtype=float) #Converts the Mat to dataframe
    MKRcolumnname = ['Time', 'R.ASISX', 'R.ASISY', 'R.ASISZ', 'L.ASISX', 'L.ASISY', 'L.ASISZ',
                     'R.PSISX', 'R.PSISY', 'R.PSISZ', 'L.PSISX', 'L.PSISY', 'L.PSISZ', 'L.Iliac.CrestX',
                     'L.Iliac.CrestY', 'L.Iliac.CrestZ', 'R.Iliac.CrestX', 'R.Iliac.CrestY', 'R.Iliac.CrestZ',
                     'R.GTRX', 'R.GTRY', 'R.GTRZ', 'R.KneeX', 'R.KneeY', 'R.KneeZ', 'R.HFX', 'R.HFY', 'R.HFZ', 'R.TTX',
                     'R.TTY', 'R.TTZ', 'R.AnkleX', 'R.AnkleY', 'R.AnkleZ', 'R.HeelX', 'R.HeelY',
                     'R.HeelZ', 'R.MT1X', 'R.MT1Y', 'R.MT1Z', 'R.MT5X', 'R.MT5Y', 'R.MT5Z', 'L.GTRX',
                     'L.GTRY', 'L.GTRZ', 'L.KneeX', 'L.KneeY', 'L.KneeZ', 'L.HFX', 'L.HFY', 'L.HFZ',
                     'L.TTX', 'L.TTY', 'L.TTZ', 'L.AnkleX', 'L.AnkleY', 'L.AnkleZ', 'L.HeelX', 'L.HeelY',
                     'L.HeelZ', 'L.MT1X', 'L.MT1Y', 'L.MT1Z', 'L.MT5X', 'L.MT5Y',
                     'L.MT5Z']  # Column Titles for the MKR dataframe
    MKRdf.columns = MKRcolumnname  # Titling the MKR dataframe
    COGX_RAW=(MKRdf['R.ASISX']+MKRdf['L.ASISX']+MKRdf['R.PSISX']+MKRdf['L.PSISX'])/4000
    COGY_RAW = (MKRdf['R.ASISY'] + MKRdf['L.ASISY'] + MKRdf['R.PSISY'] + MKRdf['L.PSISY']) / 4000
    COGZ_RAW = (MKRdf['R.ASISZ'] + MKRdf['L.ASISZ'] + MKRdf['R.PSISZ'] + MKRdf['L.PSISZ']) / 4000
    Time=pd.DataFrame(((GRFdf['Time'].iloc[-3001:])/100)-60)
    COGX_CALC=COG_LinearizationMK2(COGX_RAW,MKRdf['Time'],Time)
    COGY_CALC = COG_LinearizationMK2(COGY_RAW, MKRdf['Time'], Time)
    COGZ_CALC = COG_LinearizationMK2(COGZ_RAW, MKRdf['Time'], Time)
    Output=np.array(np.hstack((COGX_CALC,COGY_CALC,COGZ_CALC)))
    #print(Output.shape)
    Input_Data = GRFdf[['Time', 'Fx1', 'Fy1', 'Fz1', 'COPx1', 'COPz1', 'Fx2',
                        'Fy2', 'Fz2', 'COPx2', 'COPz2']]  # Input data dataframe
    Input_Data.loc[:,'Time']=Time
    Input_Data.loc[:, 'COPx1'] = Input_Data['COPx1'] / 1000
    # # Input_Data['COPy1']=Input_Data['COPy1']/1000
    Input_Data.loc[:, 'COPz1'] = Input_Data['COPz1'] / 1000
    Input_Data.loc[:, 'COPx2'] = Input_Data['COPx2'] / 1000
    # # Input_Data['COPy2']=Input_Data['COPy2']/1000
    Input_Data.loc[:, 'COPz2'] = Input_Data['COPz2'] / 1000
    height_array = np.full((len(Input_Data), 1), float(Hieght))
    mass_array = np.full((len(Input_Data), 1), float(Mass))
    Input_Data.insert(loc=Input_Data.shape[1], column="Height",value=height_array)
    Input_Data.insert(loc=Input_Data.shape[1], column="Mass", value=mass_array)
    Input_Data=Input_Data.iloc[-3001:]
    Input= Input_Data.to_numpy(dtype=np.float64)
    nan_coordinates =np.unique(np.where(np.isnan(Output))[0])
    #Output=NaN_Linearization(nan_coordinates,Time,Output)
    #print(Input.shape)
    return Input, Output
