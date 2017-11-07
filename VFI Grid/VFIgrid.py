#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:18:03 2017

@author: klp4
"""
import numpy as np
import itertools

def rouwen(rho, mu, step, num):
    '''
    Adapted from Lu Zhang and Karen Kopecky. Python by Ben Tengelsen.
    Construct transition probability matrix for discretizing an AR(1)
    process. This procedure is from Rouwenhorst (1995), which works
    well for very persistent processes.

    INPUTS:
    rho  - persistence (close to one)
    mu   - mean and the middle point of the discrete state space
    step - step size of the even-spaced grid
    num  - number of grid points on the discretized process

    OUTPUT:
    dscSp  - discrete state space (num by 1 vector)
    transP - transition probability matrix over the grid
    '''

    # discrete state space
    dscSp = np.linspace(mu -(num-1)/2*step, mu +(num-1)/2*step, num).T

    # transition probability matrix
    q = p = (rho + 1)/2.
    transP = np.array([[p**2, p*(1-q), (1-q)**2], \
                    [2*p*(1-p), p*q+(1-p)*(1-q), 2*q*(1-q)], \
                    [(1-p)**2, (1-p)*q, q**2]]).T

    while transP.shape[0] <= num - 1:

        # see Rouwenhorst 1995
        len_P = transP.shape[0]
        transP = p * np.vstack((np.hstack((transP, np.zeros((len_P, 1)))), np.zeros((1, len_P+1)))) \
                + (1 - p) * np.vstack((np.hstack((np.zeros((len_P, 1)), transP)), np.zeros((1, len_P+1)))) \
                + (1 - q) * np.vstack((np.zeros((1, len_P+1)), np.hstack((transP, np.zeros((len_P, 1)))))) \
                + q * np.vstack((np.zeros((1, len_P+1)), np.hstack((np.zeros((len_P, 1)), transP))))

        transP[1:-1] /= 2.

    # ensure columns sum to 1
    if np.max(np.abs(np.sum(transP, axis=1) - np.ones(transP.shape))) >= 1e-12:
        print('Problem in rouwen routine!')
        return None
    else:
        return transP.T, dscSp
    
    
    
def VFIsolve(funcname, Xbar, Ybar, Sigma, nx, ny, nz, npts):
    
    # set VF iteration parameters
    ccrit = 1.0E-10
    maxwhile = 1000
    

    # find sizes and shapes of functions
    XZdims = []
    for i in range(0,nx):
        XZdims.append(npts)
    for i in range(0,nz):
        XZdims.append(npts)
    
    # initialize value, policy and jump functions
    Vf1 = np.ones(XZdims) * (-100)
    Vf1new = np.zeroslike(Vf1)
    # need vecotor stored at each node.
    Pf1 = np.zeros((knpts, znpts))
    Jf1 = np.zeros((knpts, znpts))

    # set up Markov approximation of AR(1) process using Rouwenhorst method
    spread = 5.  # number of standard deviations above and below 0
    znpts = npts
    zstep = 4.*spread*sigma_z/(npts-1)
    
    # Markov transition probabilities, current z in cols, next z in rows
    Pimat, zgrid = rouwen(rho_z, 0., zstep, znpts)
    
    # discretize X variables
    Xlow = .6*Xbar
    Xhigh = 1.4*Xbar
    for i in range(0, nx):
        Xgrid = np.linspace(Xlow[i], Xhigh[i], num = npts)
    
    # discretize Y variables

    
    # run the program to get the value function (VF1)
    count = 0
    dist = 100.
    nconv = True 
    while (nconv):
        count = count + 1
        if count > maxwhile:
            break
        for i1 in range(0, knpts): # over kt
            for i2 in range(0, znpts): # over zt, searching the value for the stochastic shock
                maxval = -100000000000
                for i3 in range(0, knpts): # over k_t+1
                    for i4 in range(0, knpts): # over ell_t
                        Y, w, r, T, c, i, u = Modeldefs(kgrid[i3], kgrid[i1], \
                            ellgrid[i4], zgrid[i2], params)
                        temp = u
                        for i5 in range(0, znpts): # over z_t+1
                            temp = temp + beta * Vf1[i3,i5] * Pimat[i2,i5]
                        # print i, j, temp (keep all of them)
                        if np.iscomplex(temp):
                            temp = -1000000000
                        if np.isnan(temp):
                            temp = -1000000000
                        if temp > maxval:
                            maxval = temp
                            Vf1new[i1, i2] = temp
                            Pf1[i1, i2] = kgrid[i3]
                            Jf1[i1, i2] = ellgrid[i4]
    
        # calculate the new distance measure, we use maximum absolute difference
        dist = np.amax(np.abs(Vf1 - Vf1new))
        if dist < ccrit:
            nconv = False
        # report the results of the current iteration
        print ('iteration: ', count, 'distance: ', dist)
        
        # replace the value function with the new one
        Vf1 = 1.0*Vf1new