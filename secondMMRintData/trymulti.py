import rebound
import numpy as np
import pandas as pd
import warnings
from multiprocessing import Process
import sys
from collections import OrderedDict



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

InitialData = pd.concat([InitialDataRand, InitialDataRes])

sys.path.insert(1, '..')
#print(path)
from SPOCKalt import *
sys.path.insert(1, '../SPOCKalt')
#Intigration/simsetup.py
from SPOCKalt import featureKlassifier
from SPOCKalt import simsetup
from multiprocessing import Manager

mgr = Manager()
ns = mgr.Namespace()
ns.df = pd.DataFrame()

spock = featureKlassifier.FeatureClassifier()




def runInt (start, end, data):
    
    featureData = pd.DataFrame()
    for x in range(start,end,1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = simsetup.get_sim(x,InitialData)
            simData = spock.simToData(sim)
        temp = pd.DataFrame.from_dict(simData[0][0], orient="index").T
        temp.loc[:,'prelimStable']=simData[0][1]
        temp.loc[:,'Stable']=InitialData['Stable'].iloc[x]
        temp.loc[:,'instability_time']=InitialData['instability_time'].iloc[x]
        temp.loc[:,'shadow_instability_time']=InitialData['shadow_instability_time'].iloc[x]
        temp.loc[:,'index']=x
        featureData = pd.concat([featureData,temp], sort=False, ignore_index=True)
    featureData = featureData.set_index('index')
    #return featureData
    ns.df= pd.concat([ns.df,featureData])
    #featureData.to_csv('2MMR'+str(start)+'To'+str(end)+'Outer.csv')
    
#LEARN ABOUT multiprocessing.Pool


if __name__ == "__main__":  # confirms that the code is under main function
    coreNum = 64
    
    bound = test = np.linspace(0, 138543, num=coreNum, endpoint=True, retstep=False, dtype=int, axis=0)
    procs = []
    # instantiating process with arguments
    for x in range(coreNum-1):
        # print(name)
        proc = Process(target=runInt, args=(bound[x],bound[x+1], ns.df))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
    # print(totalDataL)
    # new = pd.concat(totalDataL)
    #ns.df = ns.df.set_index('index')
    sheetname = 'FirstMultiTest'
    ns.df.to_csv(sheetname+'.csv')
    new = pd.read_csv(sheetname+'.csv').set_index('index').sort_index()
    new.to_csv(sheetname+'.csv')
 