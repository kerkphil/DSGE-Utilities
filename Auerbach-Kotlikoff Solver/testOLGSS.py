# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 13:31:47 2017

@author: Kerk Phillips

This program solves the steady state for an S-period OLG model with endogenous
labor from an elliptical utility function for leisure.  There is a warm-glow 
bequest motive in the final period of life.  Mortality is still zero prior to 
period S, so there are no unintended bequests.
"""

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from AKsolve import AKsolve

# This is the labor-leisure Euler equation set up to be zero by definition
def LLEuler(n, c, *HHparams):
    # unpack HHparams
    wbar = HHparams[0]
    return wbar*c**(-sig) - chin*b*n**(ups-1)*(1-n**ups)**((1-ups)/ups)

   
# find the history of consumptions, labor and savings, given a value for c(1),
# wbar, rbar and model parameters
def conshist(c1, *HHparams):
    # unpack HHparams
    wbar = HHparams[0]
    rbar = HHparams[1]
    # initialize vectors for consumption, labor and savings
    chist = np.zeros(S)
    nhist = np.zeros(S)
    bhist = np.zeros(S+1)
    # set intitial consumption value
    chist[0] = c1
    for t in range(0, S):
        # set up lambda function for fsolve with n as the input
        f1 = lambda n:  LLEuler(n, chist[t], *HHparams)
        # solve LLEuler for value of n given c
        nhist[t] = opt.fsolve(f1, .99)
        if printcheck:
            # check that LLEuler is close to zero and report
            check = f1(nhist[t])
            print "nhist", t, ": ", nhist[t], " check-n: ", check
        # solve for b given c and n
        bhist[t+1] = wbar*nhist[t] + (1.+rbar)*bhist[t] - chist[t]
        # if not the final period solve for next period's c from interpemporal
        # Euler equation
        if t< S-1:
            chist[t+1] = chist[t]*(bet*(1.+rbar))**(1/sig)
    return chist, nhist, bhist
    

# find IT Euler error at death, given a value for c(1), wbar, rbar and model 
# parameters    
def findc1(c1, *HHparams):
    # This function is for fsolve it takes initial consumption and returns 
    # final IT Euler error.
    chist, nhist, bhist = conshist(c1, *HHparams)
    return chist[S-1] - chib**(-1.0/sig)*bhist[S]


# find updated wbar and rbar from initial guesses
def updatebar(bar, UPparams):
    # unpack the bar vector
    wbar = bar[0]
    rbar = bar[1]
    # set parameters to pass to findc1 in fsolve
    HHparams = (wbar, rbar, sig, chin, chib, b, ups, bet, S, printcheck)
    # solve findc1 for value of c1 
    c1opt = opt.fsolve(findc1, .1, args=HHparams)
    # check that final savings is close to zero and report
    # check = findc1(c1opt, *HHparams)
    # print "check-c: ", check, "wbar & rbar: ", wbar, rbar
    # get the consumer's full history
    chist, nhist, bhist = conshist(c1opt, *HHparams)
    # sum savings and labor to get aggregate capital annd labor inputs
    Kbar = np.sum(bhist)
    if Kbar < .01:
        Kbar = .01
    Lbar = np.sum(nhist)
    # solve for the implied values of wages and interest rates
    wbarnew = (1-alf)*(Kbar/Lbar)**alf
    rbarnew = alf*(Lbar/Kbar)**(1-alf) - delta
    # put into barnew array
    barnew = np.array([wbarnew, rbarnew])
    return barnew


# Main program follows     
# set model parameter values
S = 100         # max age
alf = .33       # capital share
bet = .99       # annual subjective discount factor
delta = .08     # annual depreciation rate
sig = 3.0       # intertemporal elasticity of substitution
chin = 10.0     # utility weight on leisure
b = 1.0         # parameter in elliptical utility
ups = 2.2926    # curvature in elliptical utility
chib = .1       # utility eight on bequest in last period of life
printcheck = 0
# set AKsolve parameter values
ccrit = 1e-12   # convegence criterion for A-K solution
conv = .25      # initital weight on new values in convexifier
maxiter = 500 # maximum number of iterations allowed in A-K solver
shrinkon = 1    # binary flag to implement shrinking conv value
shrink = .75    # factor to reduce conv if A-K not converging
expandon = 1    # binary flag to implement expanding conv value
expand = 1.01   # factor to expand conv if A-K is converging
disttype = 1    # distance measure formula
display = 1     # display iterations on or off
# convert beta and delta to per period values
bet = bet**(100./S)
delta = 1. - (1. - delta)**(100./S)
# set paramters to pass to updatebar
UPparams = (sig, chin, chib, b, ups, bet, S, printcheck)
wbar = 1
rbar = .02
# set parameters to pass to findc1 in fsolve
HHparams = (wbar, rbar, sig, chin, chib, b, ups, bet, S, printcheck)
# set initial guess for wbar and rbar
bar = np.array([wbar, rbar])

bar = AKsolve(bar, updatebar, UPparams, ccrit, conv, maxiter, shrinkon, \
    shrink, expandon, expand, disttype, display)
        
# recover wbar and rbar and print
wbar = bar[0]
rbar = bar[1]
rbar_ann = (1.+rbar)**(S/100.) - 1.
print "wbar: ", wbar, "rbar: ", rbar, "rbar annual: ", rbar_ann
# check the equations have solved
#HHparams = (wbar, rbar, sig, chin, chib, b, ups, bet, S, printcheck)
c1opt = opt.fsolve(findc1, .1, args=HHparams)
printcheck = 1
chist, nhist, bhist = conshist(c1opt, *HHparams)
# drop last period for savings (it is now zero anyway)
bhist = bhist[0:S]
# plot steady state values for consumption, savings and labor
plt.subplot(3, 1, 1)
plt.plot(chist)
plt.title('Consumption')
plt.xlabel('age')
plt.subplot(3, 1, 2)
plt.plot(nhist)
plt.title('Labor')
plt.xlabel('age')
plt.subplot(3, 1, 3)
plt.plot(bhist)
plt.title('Savings')
plt.xlabel('age')
plt.show()