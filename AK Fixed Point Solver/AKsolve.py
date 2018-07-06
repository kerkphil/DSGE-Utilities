# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 13:31:47 2017
Updated on Mon Oct 4 15:14 2017
@author: Kerk Phillips
"""
import numpy as np

def AKsolve(Xguess, funcname, fparams, ccrit, damp, maxiter, shrinkon, \
    shrink, expandon, expand, disttype, display):
    '''
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
        damp:  The weight put on Xnew relative to Xvalue when moving to the
            next iteration; Xvalue = damp*Xnew + (1-damp)*Xvalue.
        maxiter:  The maximum number of iterations allowed
        shrinkon:  If true, the value of damp is scaled down when the distance
            between values of X in an iteration increases.
        shrink:  The factor by which damp shrinks.
        expandon:  If true, the value of damp is scaled up when the distance
            between values of X in an iteration does not increase.
        expand:  The factor by which damp expands.
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
    print('Performing AK contraction mapping')
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
        # check if dist is falling, if not lower value of damp
        if (dist > distold) and (shrinkon):
            # shrink damp and redo with same Xvalue, do not update count
            damp = damp * shrink
            Xvalue = damp*Xnew + (1-damp)*Xvalue
            distold = dist
        else:
            # update Xvalue and count
            count = count + 1
            if expandon:
                # expand damp if it is < 1.0
                if damp < 1.0:
                    damp = damp * expand
                else:
                    damp = 1.0
            # take convex combination for new guess
            Xvalue = damp*Xnew + (1-damp)*Xvalue
            # replace old dist value
            distold = dist
        # show progress
        if display:
            print ('count: ', count, 'distance: ', dist, 'damp: ', damp)
    
    return Xvalue, dist, count