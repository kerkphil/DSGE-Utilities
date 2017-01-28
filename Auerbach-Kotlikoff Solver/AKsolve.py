# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 13:31:47 2017

@author: Kerk Phillips
"""
import numpy as np

def AKsolve(Xguess, funcname, fparams, ccrit, conv, maxiter, shrinkon, \
    shrink, expandon, expand, disttype, display):
    '''
    Version 1.0 January 2017
    
    This function performs the Auerbach-Kotlikoff contraction mapping on a 
    function.
    
    The inputs are:
        Xguess:  An initial guess for the fixed point. Can be a scalar or
            matrix.
        funcname:  Ahe name of the python function.  It must take Xvalue as an
            argument with the same dimensions as Xguess, with fparams as 
            parameters and return a new value for X, Xnew.        
        fparams:  A list of parameters used by funcname
        ccrit:  The value for distance between Xvalue and Xnew that indicates
            convergence to the fixed point
        conv:  The weight put on Xnew relative to Xvalue when moving to the
            next iteration; Xvalue = conv*Xnew + (1-conv)*Xvalue.
        maxiter:  The maximum number of iterations allowed
        shrinkon:  If true, the value of conv is scaled down when the distance
            between values of X in an iteration increases.
        shrink:  The factor by which conv shrinks.
        expandon:  If true, the value of conv is scaled up when the distance
            between values of X in an iteration does not increase.
        expand:  The factor by which conv expands.
        disttype:  Indicator variable for the method used to compute distance
            between Xvalue and Xnew
            1: root mean squared differences  (default)
            2: mean absolute deviation
            3: maximum absolute deviation
        display:  If true, display iterations.
    
    The outputs are the fixed point, the last iteration's distanceand the
        number of iterations performed
    '''
    # initialize Xvalue
    Xvalue = Xguess
    # set initial distance measures
    dist = 1.0
    distold = 2.0
    # set counter
    count = 0
    # begin AK iterations
    print "Performing AK contraction mapping"
    while dist > ccrit:
        if count > maxiter:
            break
        Xnew = funcname(Xvalue, fparams)
        diff = Xnew - Xvalue
        if disttype == 2:
            dist = np.mean(np.absolute(diff))
        elif disttype == 3:
            dist = np.amax(np.absolute(diff))
        else:
            dist = (np.mean(diff**2))**.5
        # check if dist is falling, if not lower value of conv
        if (dist > distold) and (shrinkon):
            # shrink conv and redo with same Xvalue, do not update count
            conv = conv * shrink
            Xvalue = conv*Xnew + (1-conv)*Xvalue
            distold = dist
        else:
            # update Xvalue and count
            count = count + 1
            if expandon:
                # expand conv if it is < 1.0
                if conv < 1.0:
                    conv = conv * 1.01
                else:
                    conv = 1.0
            # take convex combination for new guess
            Xvalue = conv*Xnew + (1-conv)*Xvalue
            # replace old dist value
            distold = dist
        # show progress
        if display:
            print "count: ", count, "distance: ", dist, "conv: ", conv
    
    return Xvalue, dist, count