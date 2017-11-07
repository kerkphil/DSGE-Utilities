# -*- coding: utf-8 -*-
"""
Version 1.0
Mon Dec 08 2016
@author: Kerk L. Phillips

This example solve and simulates a simple DSGE model by a cloded-form solution
for the steady state and linearization about that steady state for the 
transition function.  It also calculates key business cycle moments and 
Euler equation errors.  It simulates nsim times with nobs observations in each
simulation.  The simulations are run in serial.  The moments, Euler errors and 
execution time are placed in Pandas dataframes and then written to an Excel
file.

"""
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from LinApp_Deriv import LinApp_Deriv
from LinApp_Solve import LinApp_Solve
from EulerErrors import EEcalc
from DSGEmoments import calcmom
    
def example_def(kp, k, z, param):
    # calculate definitions for GDP, wages, rental rates, consumption
    np.exp((1-alpha)*z)
    y = k**alpha*np.exp((1-alpha)*z)
    w = (1-alpha)*y
    r = alpha*y/k
    c = w + (1+r-delta)*k - kp*(1+g)*(1+n)
    i = y - c
    return y, c, i, r, w

def example_dyn(invec, param):
    # unpack in
    kplus = invec[0]
    know = invec[1]
    kminus = invec[2]
    zplus = invec[3]
    znow = invec[4]

    # get definitions each period
    ynow, cnow, inow, rnow, wnow = example_def(know, kminus, znow, param)
    yplus, cplus, iplus, rplus, wplus = example_def(kplus, know, zplus, param)

    # calculate Gamma function (Euler equation)
    Gamma = cnow**(-theta) / ((cplus**(-theta)*(1+rplus-delta)/((1+g)**theta* \
                            (1+rho)))) - 1.0
    return Gamma
    
def example_efunc(kpp, kp, k, zp, z, epars):
    invec = np.array([kpp[0,0], kp[0,0], k[0], zp[0], z[0]])
    outvec = example_dyn(invec, epars)
    return outvec
 
def example_tfunc(k, z, tpars):
    kp = PP*(k-kbar) + QQ*z + kbar
    return kp
     
def example_lfunc(z, eps, lpars):
    zp = phi*z + sigma*eps
    return zp
    
def simulate_serial(nsim, nobs, ntoss, simpars):
    # initialize arrays to store simulation results
    MomentsAll = np.zeros((6, 7, nsim))
    EEmatAll = np.zeros((3, nsim))
     
    # perform simulations
    for s in range(0, nsim):
        # draw randowm errors
        eps = np.random.randn(nobs)*sigma
        # initialize variables
        z = np.zeros((nobs+1))
        k = np.zeros((nobs+1))
        y = np.zeros(nobs)
        r = np.zeros(nobs)
        w = np.zeros(nobs)
        i = np.zeros(nobs)
        c = np.zeros(nobs)
        # set starting values
        k[0] = kbar
        z[0] = eps[0]
        
        # iteratively generate data
        for t in range(0, nobs):
            z[t+1] = example_lfunc(z[t], eps[t], lpars)
            k[t+1] = example_tfunc(k[t], z[t], tpars)
            y[t], c[t], i[t], r[t], w[t] = \
                example_def(k[t+1], k[t], z[t], param)
        # discard las observation for k & z
        k = k[0:nobs];
        z = z[0:nobs];
        # stack data into a single array for moments calculation
        data = np.stack((y, c, i, r, w, k, z), axis=1)
        # discard first ntoss observations
        data = data[ntoss:nobs,:]
        
        # find moments
        (Moments, MomNames) = calcmom(data, means = True, stds = True, 
            relstds = True, corrs = True, autos = True, cvars = True)
        
        # find Euler Errors
        Xdata = k.reshape(len(k), 1)
        Zdata = z.reshape(len(z), 1)
        EErrs = EEcalc(Xdata, Zdata, efunc, epars, tfunc, tpars, lfunc, lpars)
        MaxAbsEE = np.max(np.abs(EErrs))
        MeanAbsEE = np.mean(np.abs(EErrs))
        RootMeanSqEE = (np.mean(EErrs**2))**.5
        EEmat = np.stack((MaxAbsEE, MeanAbsEE, RootMeanSqEE))
        
        # Add results of current simulation to results arrays
        MomentsAll[:, :, s] = Moments
        EEmatAll[:, s] = EEmat
    
    return MomentsAll, EEmatAll, MomNames

# set parameter values
#  model
g = .025
n = .01
delta = .08
alpha = .33
theta = 2.5
rho = .05
phi = .9
sigma = .0075
beta = (1+g)**(1-theta)*(1+n)/(1+rho)
param = [g, n, delta, alpha, theta, rho, phi, sigma, beta]

# set program paramters
nobs = 300      # number of periods in simulatipon
ntoss = 100     # number of perids to delete from begining of the simulation
nsim = 10      # number of simulations to perform

# calculate steady state values
kbar = (((1+rho)*(1+g)**theta-1+delta)/alpha)**(1/(alpha-1))
ybar = kbar**alpha
rbar = alpha*ybar/kbar
wbar = (1-alpha)*ybar
cbar = wbar + (1+rbar-delta)*kbar - (1+g)*(1+n)*kbar
ibar = ybar - cbar
reportbar = np.array([[ybar],[cbar],[ibar],[kbar],[wbar],[rbar]])

# check SS values
invec = np.array([kbar, kbar, kbar, 0, 0])
check = example_dyn(invec, param)
print 'SS check', check

# find derivatives
AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT = \
    LinApp_Deriv(example_dyn,param,invec,1,0,1,0);

# find policy function coefficients
PP, QQ, UU, RR, SS, VV = \
    LinApp_Solve(AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,WW,TT,phi,0,1);

# set function names for EE calculator
efunc = example_efunc
epars = param
tfunc = example_tfunc
tpars = (PP, QQ, kbar)
lfunc = example_lfunc
lpars = (phi)

# collect parameters for simulate_serial function
simpars = (efunc, epars, tfunc, tpars, efunc, epars)

# start timer
start = timer()
# perform simulations
(MomentsAll, EEmatAll, MomNames) = \
     simulate_serial(nsim, nobs, ntoss, simpars)
# end timer
elapsed = timer() - start  
   
# take averages over the simulations    
MomentsAvg = MomentsAll.mean(axis=2)
EEmatAvg = EEmatAll.mean(axis=1)
OtherMat = np.concatenate((EEmatAvg, np.array([elapsed])), axis = 0)
# put averages into Pandas dataframes
VarNames = ['y','c','i','r','w','k','z']
MomTab = pd.DataFrame(MomentsAvg, index = MomNames, columns = VarNames) 
OtherNames = ['MaxAbsEE', 'MeanAbsEE', 'RootMeanSqEE', 'computation time']
OtherTab = pd.DataFrame(OtherMat, index = OtherNames)

print ('moments table: ', MomTab)
print ('Euler errors: ', OtherTab)
print ('computation time', elapsed)

writer = pd.ExcelWriter('Ex1_Serial.xlsx')
MomTab.to_excel(writer,'Sheet1')
OtherTab.to_excel(writer,'Sheet2')
writer.save()