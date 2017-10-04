#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 06:08:37 2017

@author: klp4
"""

import numpy as np

def masmooth(data, lead, lag):
    
    '''
    This function calculates a moving average over a window of specified leads
    and lags.
    
    The arguments are:
    data - a numpy vector
    lead - an integer value for the number of leading values to include
    lag - an integer value for the number of lagging values to include
    
    The output is:
    smooth - a numpy vector of the same size as 'data'            
    '''
    # find number of observations
    nobs  = data.size
    
    # initialize smoothed series
    smooth = np.zeros(nobs)
    
    if nobs < lead + lag:
        print('lead plus lag must be smaller than the sample size')
        smooth[:] = np.NAN
    else:
        #calculate averages
        for i in range(0, nobs):
            if i < lag:
                smooth[i] = np.mean(data[0 : i + lead])
            elif i < (nobs - lead):
                smooth[i] = np.mean(data[i - lag : i + lead])
            else:
                smooth[i] = np.mean(data[i - lag : nobs])
            
    return smooth
            
            
# example
import matplotlib.pyplot as plt

# generate randomnumbers for data
data = np.random.rand(100)       

# smooth using the masmooth function
smooth = masmooth(data, 15, 15)

# plot data and smooth
plt.plot(data)
plt.plot(smooth)
plt.show()