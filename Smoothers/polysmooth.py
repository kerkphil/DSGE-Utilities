#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 06:39:54 2017

@author: klp4
"""

import numpy as np

def polysmooth(data, N):
    
    '''
    This function calculates a polynomial fit for data over time.
    
    The arguments are:
    data - a numpy vector
    N - an integer value for the order of the polynomial
    
    The output is:
    smooth - a numpy vector of the same size as 'data'            
    '''
    # find number of observations
    nobs  = data.size
    
    # initialize smoothed series
    smooth = np.zeros(nobs)
    
    if nobs < N:
        print('N must be smaller than the sample size')
        smooth[:] = np.NAN
    else:
        # construct X matrix
        T = np.linspace(0, nobs-1, nobs)
        X = np.ones((nobs,1))
        for i in range(1, N):
            X = np.hstack((X, np.reshape(T**i, (-1, 1))))
        
        # calculate beta
        X1 = np.linalg.inv(np.dot(np.transpose(X), X))
        X2 = np.dot(np.transpose(X), data)
        beta = np.dot(X1, X2)
        beta = np.reshape(beta, (-1, 1))
        
        # get fitted values
        smooth = np.dot(X, beta) 
        
        smooth = smooth.flatten()
            
    return smooth
            
            
# example
import matplotlib.pyplot as plt

# generate randomnumbers for data
data = np.random.rand(100)       

# smooth using the masmooth function
smooth = polysmooth(data, 9)

# plot data and smooth
plt.plot(data)
plt.plot(smooth)
plt.show()