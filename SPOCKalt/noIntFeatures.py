from collections import OrderedDict
import numpy as np
class Trio:
    def __init__(self):
        '''initializes new set of features.
        
            each list of the key is the series of data points, second dict is for final features
        
        '''
    #innitialize running list 
        self.runningList = OrderedDict()
        self.runningList['time']=[]
        self.runningList['MEGNO']=[]
        self.runningList['threeBRfill']=[]
        for each in ['near','far','outer']:
            self.runningList['EM'+each]=[]
            self.runningList['EP'+each]=[]
            self.runningList['MMRstrength'+each]=[]
            self.runningList['twoMMRstrength'+each]=[]
            self.runningList['MMRstrengthW'+each]=[]
            self.runningList['MMRinWid'+each]=[]
            self.runningList['twoMMRstrengthW'+each]=[]
            self.runningList['twoMMRinWid'+each]=[]
    #returned features
        self.features = OrderedDict()

        for each in ['near','far','outer']:
            self.features['EMcross'+each]= np.nan
            self.features['EMfracstd'+each]= np.nan
            self.features['EPstd'+each]= np.nan
            self.features['MMRstrength'+each]= np.nan
            self.features['twoMMRstrength'+each]= np.nan
            self.features['MMRinWid'+each]=np.nan
            self.features['MMRstrengthW'+each]=np.nan
            self.features['MMRstrengthWMAX'+each]=np.nan
            self.features['twoMMRinWid'+each]=np.nan
            self.features['twoMMRstrengthW'+each]=np.nan
            self.features['twoMMRstrengthWMAX'+each]=np.nan

        self.features['MEGNO']= np.nan
        self.features['MEGNOstd']= np.nan
        self.features['threeBRfillfac']= np.nan
        self.features['threeBRfillstd']= np.nan
        self.features['chiSec'] = np.nan

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
        
        for q, [label, i1, i2] in enumerate(pairs):
            m1 = ps[i1].m
            m2 = ps[i2].m
            e1x, e1y = ps[i1].e*np.cos(ps[i1].pomega), ps[i1].e*np.sin(ps[i1].pomega)
            e2x, e2y = ps[i2].e*np.cos(ps[i2].pomega), ps[i2].e*np.sin(ps[i2].pomega)
            self.runningList['time'][i]= sim.t/minP
            self.runningList['EM'+label][i]= np.sqrt((e2x-e1x)**2 + (e2y-e1y)**2)
            self.runningList['EP'+label][i] = np.sqrt((m1*e1x + m2*e2x)**2 + (m1*e1y + m2*e2y)**2)/(m1+m2)
            MMRs = find_strongest_MMR(sim, i1, i2)
            self.runningList['MMRstrength'+label][i] = MMRs[2]
            self.runningList['twoMMRstrength'+label][i] = MMRs[6]
            MMRW = MMRwidth(sim, MMRs[3], i1,i2)
            self.runningList['MMRstrengthW'+label][i]=MMRW[0]
            self.runningList['MMRinWid'+label][i]=MMRW[1]
            MMRWtwo = MMRwidth(sim, MMRs[7], i1,i2)
            self.runningList['twoMMRstrengthW'+label][i]=MMRWtwo[0]
            self.runningList['twoMMRinWid'+label][i]=MMRWtwo[1]
        self.runningList['threeBRfill'][i]= threeBRFillFac(sim, trio)
        self.runningList['MEGNO'][i]= sim.megno() 

    def startingFeatures(self, sim, pairs):
        '''used to initialize/add to the features that only depend on initial conditions'''
        
        #only applies to one
        ps  = sim.particles
        for [label, i1, i2] in pairs:
            self.features['EMcross'+label] =  (ps[i2].a-ps[i1].a)/ps[i1].a
        

    
        #FIXME
        #this wont work for abstracted data but will be used for testing purposes
        #from Eritas&Tamayo 2024

        #equation 11
        e12 = 1- (ps[1].a/ps[2].a)
        e13 = 1- (ps[1].a/ps[3].a)
        e23 = 1- (ps[2].a/ps[3].a)

        #23
        eta = (e12/e13)-(e23/e13)

        #equation 25

        mu = (ps[3].m-ps[1].m)/(ps[1].m+ps[3].m)
        chi23 = (1+eta)**3 *(3-eta)*(1+mu)
        chi12 = (1-eta)**3 *(3+eta)*(1-mu)

        self.features['chiSec']= chi12/(chi23+chi12)

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

        self.features['threeBRfillfac']= np.median(self.runningList['threeBRfill'])
        self.features['threeBRfillstd']= np.std(self.runningList['threeBRfill'])


        for label in ['near', 'far', 'outer']: #would need to remove outer here
            self.features['MMRstrength'+label] = np.median(self.runningList['MMRstrength'+label])
            self.features['twoMMRstrength'+label]= np.median(self.runningList['twoMMRstrength'+label])
            self.features['EMfracstd'+label]= np.std(self.runningList['EM'+label])/ self.features['EMcross'+label]
            self.features['EPstd'+label]= np.std(self.runningList['EP'+label])
            self.features['MMRstrengthW'+label]=np.median(self.runningList['MMRstrengthW'+label])
            self.features['MMRstrengthWMAX'+label]=max(self.runningList['MMRstrengthW'+label])
            self.features['twoMMRstrengthW'+label]=np.median(self.runningList['twoMMRstrengthW'+label])
            self.features['twoMMRstrengthWMAX'+label]=max(self.runningList['twoMMRstrengthW'+label])
            self.features['MMRinWid'+label]=np.median(self.runningList['MMRinWid'+label])
            self.features['twoMMRinWid'+label]=np.median(self.runningList['twoMMRinWid'+label])
       
def MMRwidth(sim,Prat, i1,i2):
    '''calculates the MMR width per tamayo&hadden 2024 equation 19
    
    returns dP/P'''
    if Prat is np.nan or type(Prat) == float or np.nan in Prat:
        #print(Prat)
        return 0, 0
    

    ps = sim.particles

    Ak = [0., 0.84427598, 0.75399036, 0.74834029, 0.77849985, 0.83161366] #from tamayo&hadden 2024
    a,b=Prat

    ec = (ps[i2].a-ps[i1].a)/ps[i2].a #crossing excentricity

    mu = (ps[i1].m+ps[i2].m)/ps[0].m #planet mass star mass ratio

    order = b-a

    erel = [ps[i2].e*np.cos(ps[i2].pomega)-ps[i1].e*np.cos(ps[i1].pomega),ps[i2].e*np.sin(ps[i2].pomega)-ps[i1].e*np.sin(ps[i1].pomega)]

    etilde = np.linalg.norm(erel)/ec

    dP = 3*Ak[order]*np.sqrt(mu * (etilde**order))

    realPrat = ps[i1].P/ps[i2].P
    inwidth = dP< abs(realPrat-(a/b))
    scaleWidth = dP/(a/b)
    

    return scaleWidth, inwidth


def threeBRFillFac(sim, trio):
    '''calculates the 3BR filling factor in acordance to petit20'''
    ps = sim.particles
    b0, b1,b2,b3 = ps[0], ps[trio[0]], ps[trio[1]], ps[trio[2]]
    m0,m1,m2,m3 = b0.m,b1.m,b2.m,b3.m
    ptot = None

    #semim
    a12 =(b1.a/b2.a)
    a23 = (b2.a/b3.a)

    #equation 43
    d12 = 1- a12
    d23 = 1- a23

    #equation 45
    d = (d12*d23)/(d12+d23)

    #equation 19
    mu12 = b1.P/b2.P
    mu23 = b2.P/b3.P

    #equation 21
    eta = (mu12*(1-mu23))/(1-(mu12*mu23))

    #equation 53
    eMpow2 = (m1*m2 + m2*m3*(a12**(-2))+m1*m2*(a23**2)*((1-eta)**2))/(m0**2)

    #equation 59
    dov = ((42.9025)*(eMpow2)*(eta*((1-eta)**3)))**(0.125)

    #equation 60

    ptot = (dov/d)**4

    return abs(ptot)






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
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    minperiodratio = max(Pratio-delta, 0.)
    maxperiodratio = min(Pratio+delta, 0.99) # too many resonances close to 1
    res = resonant_period_ratios(minperiodratio,maxperiodratio, order=maxorder)

    # Calculating EM exactly would have to be done in celmech for each j/k res below, and would slow things down. This is good enough for approx expression
    EM = np.sqrt((ps[i1].e*np.cos(ps[i1].pomega) - ps[i2].e*np.cos(ps[i2].pomega))**2 + (ps[i1].e*np.sin(ps[i1].pomega) - ps[i2].e*np.sin(ps[i2].pomega))**2)
    EMcross = (ps[i2].a-ps[i1].a)/ps[i1].a

    j, k, maxstrength, res1 = np.nan, np.nan, 0, np.nan 
    j2, k2, maxstrength2, res2 = np.nan, np.nan, 0, np.nan 
    
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
            res2 = [a,b]
            if maxstrength2> maxstrength:
                j,j2 = swap(j,j2)
                k,k2 = swap(k,k2)
                res1, res2 = swap(res1,res2)
                maxstrength, maxstrength2 = swap(maxstrength, maxstrength2)
    

    # if maxstrength == 0:
    #     maxstrength = np.nan
    # if maxstrength2 == 0:
    #     maxstrength2 = np.nan

    return j, k, maxstrength, res1, j2, k2, maxstrength2, res2, res
#############################################

def swap(a,b):
    return b,a