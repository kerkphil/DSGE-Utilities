# -*- coding: utf-8 -*-
"""
Version 1.0
Fri Nov 25 2016
@author: Kerk L. Phillips
This program calculates Euler Errors from a DSGE model.  EEcalc is the function.  
The rest of the code implements this for a simple RBC model.

"""
import numpy as np
from scipy.stats import norm
from LinApp_Deriv import LinApp_Deriv
from LinApp_Solve import LinApp_Solve

def EEcalc(Xdata, Zdata, efunc, epars, tfunc, tpars, lfunc, lpars):
    '''
    This function calculates Euler Errors for a DSGE model.
    It takes the following as inputs:
    1)  Xdata: an nobs-by-nx matrix of endogenous state variable data with with 
        nobs indexing the observations in rows and nx indexing the variables in 
        columns.
    2)  Zdata: an nobs-by-nz matrix of exogenous state variable data with with 
        nobs indexing the observations in rows and nz indexing the variables in 
        columns.
    3)  efunc: the name of a function which takes a single observation of data 
        and returns the Euler error for a given realization of future exogenous 
        variables.  This function must take 3 nx-element vectora of endogenous
        state variables and and 2 nz-element vector of exogenous state 
        variables as inputs and output an neq-element vector of Euler equation 
        errors.  The order of input is X(t+2), X(t+1), X(t), Z(t+1), Z(t),
        epars
    4)  epars: a list of parameters passed to efunc.
    5)  tfunc: the name of a transition function which takes a single 
        observation of the state variables and returns next period's value of 
        the endogenous state variables.  This function must take an nx-element 
        vector of endogenous state variables and and nz-element vector of 
        exogenous state variables as inputs and output an nz-element vector of 
        next-period endogenous state variables.  The order of inputs is X(t), 
        Z(t), tpars
    6)  tpars: a list of parameters passed to tfunc.
    7)  lfunc: the name of a law-of-motion function which takes a single 
        observation of the exogenous state variables and returns next period's 
        value of the exogenous state variables.  This function must take an 
        nz-element of exogenous state and an scalar iid shock as inputs 
        and output an nz-element vector of next-period endogenous state 
        variables. The order of inputs is Z(t), Eps(t+1), epars
    8)  lpars: a list of parameters passed to lfunc.
    
    The function returns the following outputs:
    1)  Eerr: an nobs-by-neq matrix of Euler errors with nobs indexing the 
        observations in rows and neq indexing the elements from the function 
        efunc in columns.
        
    Notes:
    Xdata and Zdata must have the same number of rows.
    Neither Xdata nor Zdatamay have missing, nan, or complex values. 
    Innovations to the law of motion are drawn from a tandard normal 
    distribution.
    Currently this function only works with one innovation shock, i.e. ne=1

    To Do:
    1) Allow for more than one shock process.  May require the use of sparse grids 
    for quadrature.
    2) Use a more efficient quadrature method.  Gaussian?
    '''
    
    # set parameter values
    npts = 10 # number of point for rectangular quadrature
    
    # check sizes of data matrices
    (Xnobs, nx) = Xdata.shape
    (Znobs, nz) = Zdata.shape
    if Xnobs == Znobs:
        nobs = Xnobs
    else:
        print 'Data matrices have different numbers of observations'
        nobs = min(Xnobs, Znobs)
        Xdata = Xdata[0:nobs]
        Zdata = Zdata[0:nobs]
    
    # generate discret support for epsilon to be used in Euler error
    # Eps are the central values
    # Phi are the associated probabilities
    Eps = np.zeros(npts);
    Cum = np.linspace(0.0, 1.0, num=npts+1)+.5/npts
    Cum = Cum[0:npts]
    Phi = np.ones(npts)/npts
    Eps = norm.ppf(Cum)

    neq = nx
    
    # initialize matrix of Euler errors
    Eerr = np.zeros((nobs,neq))
    # begin loop over time periods
    for t in range(0, nobs):
        # begin loop over possible va,lues of shock next period
        for i in range(0, npts):
            # find value of next period Z
            Zp = lfunc(Zdata[t,:],Eps[i],lpars)
            # find the value of X next period
            Xp = tfunc(Xdata[t,:],Zdata[t,:],tpars)
            # find the value of X in two periods
            Xpp = tfunc(Xp,Zp,tpars)
            # find the Euler errors
            Eerr[t,:] = Eerr[t,:] + Phi[i]*efunc(Xpp,Xp,Xdata[t,:], \
                Zp,Zdata[t,:],epars)
    return Eerr

'''
All the code below is for demonstrating how to implement the EEcalc function 
for a simple RBC model.
'''
    
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

    # calculate Gamma function
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

# demonstrate the EEcalc function for a simple RBC model

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

# program
nobs = 300      # number of periods in simulatipon
kstart = 1      # starting value for simulation (proportional to kbar)
solve = 1       # set to 1 to compute coeffsd, 0 if coeffs already in memory


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

# perform simulation
tpars = (PP, QQ, kbar)
lpars = (phi)
eps = np.random.randn(nobs)*sigma
z = np.zeros((nobs+1))
k = np.zeros((nobs+1))
y = np.zeros(nobs)
r = np.zeros(nobs)
w = np.zeros(nobs)
i = np.zeros(nobs)
c = np.zeros(nobs)
k[0] = kbar*kstart
z[0] = eps[0]
for t in range(0, nobs):
    z[t+1] = example_lfunc(z[t], eps[t], lpars)
    k[t+1] = example_tfunc(k[t], z[t], tpars)
    y[t], c[t], i[t], r[t], w[t] = example_def(k[t+1], k[t], z[t], param)
k = k[0:nobs];
z = z[0:nobs];

# find Euler Errors
Xdata = k.reshape(len(k), 1)
Zdata = z.reshape(len(z), 1)
efunc = example_efunc
epars = param
tfunc = example_tfunc
lfunc = example_lfunc
EErrs = EEcalc(Xdata, Zdata, efunc, epars, tfunc, tpars, lfunc, lpars)

MaxAbsEE = np.max(np.abs(EErrs))
MeanAbsEE = np.mean(np.abs(EErrs))
RootMeanSqEE = (np.mean(EErrs**2))**.5

print 'Euler Error Summary Statistics'
print 'Maximum Absolute Euler Error:', MaxAbsEE
print 'Mean Absolute Euler Error:', MeanAbsEE
print 'Root Mean Squared Euler Error:', RootMeanSqEE