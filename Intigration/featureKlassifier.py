from features import *
from tseries import *
from simsetup import *

def simToData(sim):
    #tseries, stable = get_tseries(sim, args)
    if isinstance(sim, rebound.Simulation):
        sim = [sim]
        
    #args = []
    if len(set([s.N_real for s in sim])) != 1:
        raise ValueError("If running over many sims at once, they must have the same number of particles")
    for s in sim:
        s = s.copy()
        init_sim_parameters(s)
        #minP = np.min([p.P for p in s.particles[1:s.N_real]])
        
        check_errors(s)
        trios = [[j,j+1,j+2] for j in range(1,s.N_real-2)] # list of adjacent trios   
        featureargs = [10000, 80, trios]
        #args.append(featureargs)
        #print(args)
        print(runSim(s,featureargs))


def runSim(sim, args):
    triotseries, stable = get_tseries(sim, args)
    #calculate final vals
    for each in triotseries:
        each.fill_features(args)
    dataList = []
    for each in triotseries:
        dataList.append(each.features)
    dataList.append(stable)
    return dataList



def check_errors(sim):
    if sim.N_real < 4:
        raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets")