import rebound
import numpy as np
import pandas as pd
import warnings
#from subprocess import call
#import os
import sys
#import warnings
#warnings.filterwarnings('ignore') # to suppress warnings about REBOUND versions that I've already tested
from collections import OrderedDict
#sys.path.append(os.path.join(sys.path[0], '..'))
#from training_data_functions import gen_training_data
#from feature_functions import features
#from additional_feature_functions import additional_features

start = 75000
end = start+5000

col = ['p0m','p0x','p0y','p0z','p0vx','p0vy','p0vz','p1m','p1x','p1y','p1z','p1vx','p1vy','p1vz','p2m','p2x','p2y','p2z','p2vx','p2vy','p2vz','p3m','p3x','p3y','p3z','p3vx','p3vy','p3vz']

path = '../'

#combines labels and initial conditions, as well as gives columns
#/home/ethadhani/SPOCKalt/csvs/random/labels.csv
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
# from Intigration import features
# from Intigration import tseries

# from Intigration import featureKlassifier

# from simsetup import *
# from featureKlassifier import *



featureData = pd.DataFrame()

spock = featureKlassifier.FeatureClassifier()
for x in range(start,end,1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = simsetup.get_sim(x,InitialData)
        simData = spock.simToData(sim)
    temp = pd.DataFrame.from_dict(simData[0][0], orient="index").T
    temp.loc[:,'prelimStable']=simData[0][1]
    temp.loc[:,'Stable']=InitialData['Stable'].iloc[x]
    temp.loc[:,'index']=x
    featureData = pd.concat([featureData,temp], sort=False, ignore_index=True)
featureData = featureData.set_index('index')
featureData.to_csv('2MMR'+str(start)+'To'+str(end)+'Outer.csv')
 