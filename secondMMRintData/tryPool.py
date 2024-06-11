import rebound
import numpy as np
import pandas as pd
import warnings
from multiprocessing import Process
import sys
from collections import OrderedDict
from multiprocessing import Pool



col = ['p0m','p0x','p0y','p0z','p0vx','p0vy','p0vz','p1m','p1x','p1y','p1z','p1vx','p1vy','p1vz','p2m','p2x','p2y','p2z','p2vx','p2vy','p2vz','p3m','p3x','p3y','p3z','p3vx','p3vy','p3vz']

path = '../'
InitialRandLabels = pd.read_csv(path+"csvs/random/labels.csv", index_col = 0)
rawInitialRand = pd.read_csv(path+"csvs/random/initial_conditions.csv",header=None)
rawInitialRand.columns = col
InitialDataRand = pd.DataFrame.join(InitialRandLabels.loc[:,'runstring'],rawInitialRand) # InitialLabels.loc[:,['instability_time','shadow_instability_time','Stable']])
InitialDataRand = pd.DataFrame.join(InitialDataRand, InitialRandLabels.loc[:,['instability_time','shadow_instability_time','Stable']])

InitialResLabels = pd.read_csv(path+"csvs/resonant/labels.csv", index_col = 0)
rawInitialRes = pd.read_csv(path+"csvs/resonant/initial_conditions.csv",header=None)
rawInitialRes.columns = col
InitialDataRes = pd.DataFrame.join(InitialResLabels.loc[:,'runstring'],rawInitialRes) # InitialLabels.loc[:,['instability_time','shadow_instability_time','Stable']])
InitialDataRes = pd.DataFrame.join(InitialDataRes, InitialResLabels.loc[:,['instability_time','shadow_instability_time','Stable']])

InitialData = pd.concat([InitialDataRand, InitialDataRes]).reset_index(drop=True)
#InitialData.reset_index(inplace = True)

sys.path.insert(1, '..')
#print(path)
from SPOCKalt import *
sys.path.insert(1, '../SPOCKalt')
#Intigration/simsetup.py
from SPOCKalt import featureKlassifier
from SPOCKalt import simsetup




spock = featureKlassifier.FeatureClassifier()

# for i in range(len(df)):
#     print(df.loc[i, "Name"], df.loc[i, "Age"])

def initialize(dataset):
    datalist = []
    #dataset.shape[0]
    for row in range(dataset.shape[0]):
        data = {'Stable': [InitialData['Stable'].iloc[row]],
                'instability_time': [InitialData['instability_time'].iloc[row]],
                'shadow_instability_time': [InitialData['shadow_instability_time'].iloc[row]],
                'index': [row]
                }
        temp = pd.DataFrame(data)
        # temp.loc[:,'Stable']=InitialData['Stable'].loc[row]
        # temp.loc[:,'instability_time']=InitialData['instability_time'].iloc[row]
        # temp.loc[:,'shadow_instability_time']=InitialData['shadow_instability_time'].iloc[row]
        # temp.loc[:,'index']=row
        datalist.append((dataset.loc[row][1:29], temp))
    return datalist

def runInt (set):
    row = set[0]
    initial = set[1]
    featureData = pd.DataFrame()
   
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = simsetup.get_simList(row)
        simData = spock.simToData(sim, True)
        temp = pd.DataFrame.from_dict(simData[0][0], orient="index").T
    temp.loc[:,'prelimStable']=simData[0][1]
    temp = pd.DataFrame.join(temp,initial)

    
    #featureData = pd.concat([featureData,temp], sort=False, ignore_index=True)
    temp = temp.set_index('index')
    return temp
    


if __name__ == "__main__":  # confirms that the code is under main function

    datalist = initialize(InitialData)

    print('part one')
    #print(datalist[0])
    #138543
    
    #bound = test = np.linspace(0, 138543, num=138544, endpoint=True, retstep=False, dtype=int, axis=0)
    with Pool(63) as p:
        new = list(p.map(runInt, datalist))
    # test = list(map(runInt, bound))
    # print(test)
    sheetname = 'testInitialcondSim'
    #print(bound)
    pd.concat(new).to_csv(sheetname+'.csv')
    new = pd.read_csv(sheetname+'.csv').set_index('index').sort_index()
    new.to_csv(sheetname+'.csv')
 