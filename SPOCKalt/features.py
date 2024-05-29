from collections import OrderedDict
import numpy as np
class Trio:
    def __init__(self):
        '''initializes new set of features.
        
            each list of the key is the series of data points, second dict is for final features
        
        '''
    #innitialize running list 
    #delete all outers
        self.runningList = OrderedDict()
        self.runningList['time']=[] #must keep time
        self.runningList['EMnear']=[]
        self.runningList['EPnear']=[]
        self.runningList['EMfar']=[]
        self.runningList['EPfar']=[]
        self.runningList['EMouter']=[]
        self.runningList['EPouter']=[]
        self.runningList['MMRstrengthfar']=[]
        self.runningList['MMRstrengthnear']=[]
        self.runningList['MMRstrengthouter']=[]
        self.runningList['twoMMRstrengthfar']=[]
        self.runningList['twoMMRstrengthnear']=[]
        self.runningList['twoMMRstrengthouter']=[]
        self.runningList['MEGNO']=[]


    #innitialize final values
    #delete all outers
        self.features = OrderedDict()
        self.features['EMcrossnear']= np.nan
        self.features['EMfracstdnear']= np.nan
        self.features['EPstdnear']= np.nan
        self.features['MMRstrengthnear']= np.nan
        self.features['twoMMRstrengthnear']= np.nan
        self.features['EMcrossfar']= np.nan
        self.features['EMfracstdfar']= np.nan
        self.features['EPstdfar']= np.nan
        self.features['MMRstrengthfar']= np.nan
        self.features['twoMMRstrengthfar']= np.nan
        self.features['EMcrossouter']= np.nan
        self.features['EMfracstdouter']= np.nan
        self.features['EPstdouter']= np.nan
        self.features['MMRstrengthouter']= np.nan
        self.features['twoMMRstrengthouter']= np.nan
        self.features['MEGNO']= np.nan
        self.features['MEGNOstd']= np.nan




    



    def fillVal(self, Nout):
        '''fills with nan values
        
            Arguments: 
                Nout: number of datasets collected'''
        for each in self.runningList.keys():
            self.runningList[each] = [np.nan]*Nout

    def getNum(self):
        '''returns number of features collected as ran'''
        return len(self.runningList.keys())

    def populateData(self, sim, trio, pairs, minP,i):
        '''populates the runningList data dictionary for one time step.
        
            user must specify how each is calculated and added
        '''
        ps = sim.particles
        #print(pairs)
        for q, [label, i1, i2] in enumerate(pairs):
            #print(q)
            m1 = ps[i1].m
            m2 = ps[i2].m
            e1x, e1y = ps[i1].e*np.cos(ps[i1].pomega), ps[i1].e*np.sin(ps[i1].pomega)
            e2x, e2y = ps[i2].e*np.cos(ps[i2].pomega), ps[i2].e*np.sin(ps[i2].pomega)
            self.runningList['time'][i]= sim.t/minP
            self.runningList['EM'+label][i]= np.sqrt((e2x-e1x)**2 + (e2y-e1y)**2)
            self.runningList['EP'+label][i] = np.sqrt((m1*e1x + m2*e2x)**2 + (m1*e1y + m2*e2y)**2)/(m1+m2)
            #print([label, i1, i2])
            MMRs = find_strongest_MMR(sim, i1, i2)
            self.runningList['MMRstrength'+label][i] = MMRs[2]
            self.runningList['twoMMRstrength'+label][i] = MMRs[5]

        self.runningList['MEGNO'][i]= sim.megno() # megno


    def startingFeatures(self, sim, pairs):
        '''used to initialize/add to the features that only depend on initial conditions'''
        
        #only applies to one
        ps  = sim.particles
        for [label, i1, i2] in pairs:
            self.features['EMcross'+label] =  (ps[i2].a-ps[i1].a)/ps[i1].a

    def fill_features(self, args):
        '''fills the final set of features that are returned to the ML model.
            
            Each feature is filled depending on some combination of runningList features and initial condition features
        '''
        Norbits = args[0]
        Nout = args[1]
        trios = args[2] #
        #print(args)

        if not np.isnan(self.runningList['MEGNO']).any(): # no nans
            self.features['MEGNO']= np.median(self.runningList['MEGNO'][-int(Nout/10):]) # smooth last 10% to remove oscillations around 2
            self.features['MEGNOstd']= np.std(self.runningList['MEGNO'][int(Nout/5):])


        for label in ['near', 'far', 'outer']: #would need to remove outer here
            self.features['MMRstrength'+label] = np.median(self.runningList['MMRstrength'+label])
            self.features['twoMMRstrength'+label]= np.median(self.runningList['twoMMRstrength'+label])
            self.features['EMfracstd'+label]= np.std(self.runningList['EM'+label])/ self.features['EMcross'+label]
            self.features['EPstd'+label]= np.std(self.runningList['EP'+label])




        # self.features['MMRstrengthnear']= np.median(self.runningList['MMRstrengthnear'])
        # self.features['MMRstrengthfar']= np.median(self.runningList['MMRstrengthfar'])
        # self.features['twoMMRstrengthnear']= np.median(self.runningList['twoMMRstrengthnear'])
        # self.features['twoMMRstrengthfar']= np.median(self.runningList['twoMMRstrengthfar'])

        # self.features['EMfracstdnear']= np.std(self.runningList['EMnear'])/ self.features['EMcrossnear']
        # self.features['EMfracstdfar']= np.std(self.runningList['EMfar'])/ self.features['EMcrossfar']
        # self.features['EPstdnear']= np.std(self.runningList['EPnear'])
        # self.features['EPstdfar']= np.std(self.runningList['EPfar'])
       
        
    




























    ######################### Taken from celmech github.com/shadden/celmech
def farey_sequence(n):
    """Return the nth Farey sequence as order pairs of the form (N,D) where `N' is the numerator and `D' is the denominator."""
    a, b, c, d = 0, 1, 1, n
    sequence=[(a,b)]
    while (c <= n):
        k = int((n + b) / d)
        a, b, c, d = c, d, (k*c-a), (k*d-b)
        sequence.append( (a,b) )
    return sequence
def resonant_period_ratios(min_per_ratio,max_per_ratio,order):
    """Return the period ratios of all resonances up to order 'order' between 'min_per_ratio' and 'max_per_ratio' """
    if min_per_ratio < 0.:
        raise AttributeError("min_per_ratio of {0} passed to resonant_period_ratios can't be < 0".format(min_per_ratio))
    if max_per_ratio >= 1.:
        raise AttributeError("max_per_ratio of {0} passed to resonant_period_ratios can't be >= 1".format(max_per_ratio))
    minJ = int(np.floor(1. /(1. - min_per_ratio)))
    maxJ = int(np.ceil(1. /(1. - max_per_ratio)))
    res_ratios=[(minJ-1,minJ)]
    for j in range(minJ,maxJ):
        res_ratios = res_ratios + [ ( x[1] * j - x[1] + x[0] , x[1] * j + x[0]) for x in farey_sequence(order)[1:] ]
    res_ratios = np.array(res_ratios)
    msk = np.array( list(map( lambda x: min_per_ratio < x[0]/float(x[1]) < max_per_ratio , res_ratios )) )
    return res_ratios[msk]
##########################

# sorts out which pair of planets has a smaller EMcross, labels that pair inner, other adjacent pair outer
# returns a list of two lists, with [label (near or far), i1, i2], where i1 and i2 are the indices, with i1 
# having the smaller semimajor axis

#taken from original spock
####################################################
def find_strongest_MMR(sim, i1, i2):
    #originally 2, trying with 5th order now
    maxorder = 5
    ps = sim.particles
    n1 = ps[i1].n
    n2 = ps[i2].n

    m1 = ps[i1].m/ps[0].m
    m2 = ps[i2].m/ps[0].m

    Pratio = n2/n1
    #next want to try not to abreviate to closest

    delta = 0.03
    if Pratio < 0 or Pratio > 1: # n < 0 = hyperbolic orbit, Pratio > 1 = orbits are crossing
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    minperiodratio = max(Pratio-delta, 0.)
    maxperiodratio = min(Pratio+delta, 0.99) # too many resonances close to 1
    res = resonant_period_ratios(minperiodratio,maxperiodratio, order=maxorder)

    # Calculating EM exactly would have to be done in celmech for each j/k res below, and would slow things down. This is good enough for approx expression
    EM = np.sqrt((ps[i1].e*np.cos(ps[i1].pomega) - ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].e*np.sin(ps[i1].pomega) - ps[i2].e*np.sin(ps[i2].pomega))**2)
    EMcross = (ps[i2].a-ps[i1].a)/ps[i1].a

    j, k, maxstrength = np.nan, np.nan, 0 
    j2, k2, maxstrength2 = np.nan, np.nan, 0 
    for a, b in res:
        nres = (b*n2 - a*n1)/n1
        if nres == 0:
            s = np.inf # still want to identify as strongest MMR if initial condition is exatly b*n2-a*n1 = 0
        else:
            s = np.abs(np.sqrt(m1+m2)*(EM/EMcross)**((b-a)/2.)/nres)
        
        if s > maxstrength2 and not np.isnan(s) :
            j2 = b
            k2 = b-a
            maxstrength2 = s
            if maxstrength2> maxstrength:
                j,j2 = swap(j,j2)
                k,k2 = swap(k,k2)
                maxstrength, maxstrength2 = swap(maxstrength, maxstrength2)
    

    # if maxstrength == 0:
    #     maxstrength = np.nan
    # if maxstrength2 == 0:
    #     maxstrength2 = np.nan

    return j, k, maxstrength, j2, k2, maxstrength2
#############################################

def swap(a,b):
    return b,a




#def createFeatures(sim, n_jobs=-1):


