"""
Created November, 2015

Author:  Kerk Phillips

Based on code by Spencer Lyon

Created for the BYU Macroeconomics and Computational Lab Python Tools Library by:
    David Spencer: Brigham Young University
    Kerk Phillips: Brigham Young University
    Rick Evans: Brigham Young University
    Ben Tengensen: Carnegie Mellon University
    Bryan Perry: Massachusetts Institute of Technology

MatLab code by Kerk P. (2013) was referenced in creating this file.
"""
from __future__ import division
from numpy import tile, array, empty, ndarray, zeros, log, asarray
import numpy as np

def LinApp_Deriv(funcname,param,theta0,nx,ny,nz,logX):
    """
    This function computes the matricies AA-MM in the log-linearizatin of
    the equations in the function 'func'.

    Parameters
    ----------
    func: function
        The function that generates a vector from the dynamic equations that are
        to be linearized. This must be written to evaluate to zero in the
        steady state. Often these equations are the Euler equations defining
        the model

    theta0: array, dtype=float
        A vector of steady state values for state parameters. Place the values
        of X in the front, then the Y values, followed by the Z's.

    nx: number, dtype=int
        The number of elements in X

    ny: number, dtype=nt
        The number of elements in Y

    nz: number, dtype=int
        The number of elements in Z

    logX: binary, dtype=int
        true if log-linearizing the X & Y variables, false for simple linearization

    Returns
    -------
    AA - MM : 2D-array, dtype=float:
        The equaitons described by Uhlig in the log-linearization.
    """
    theta0 = asarray(theta0)
    length = 3 * nx + 2 * (ny + nz)
    height = nx + ny
    #eps = 2.2E-16  # machine epsilon for double-precision
    eps = 10E-6
    incr = np.zeros_like(theta0) + eps
    dx = np.zeros_like(incr) + 2.0 * incr

    T0 = funcname(theta0, param)  # should be very close to zero if linearizing about SS
    devplus = tile(theta0.reshape(1, theta0.size), (length, 1))
    devminus = tile(theta0.reshape(1, theta0.size), (length, 1))

    for i in range(length):
        devplus[i, i] += incr[i]
        devminus[i, i] -= incr[i]

    bigplus = empty((height, length))
    bigminus = empty((height, length))
    for i in range(0,length):
        if i < 3 * nx + 2 * ny:
            if logX:
                bigplus[:, i] = theta0[i] * (funcname(devplus[i, :], param) - T0) / (1.0 + T0)
                bigminus[:, i] = theta0[i] * (funcname(devminus[i, :], param) - T0) / (1.0 + T0)
            else:
                bigplus[:, i] = funcname(devplus[i, :], param)
                bigminus[:, i] = funcname(devminus[i, :], param)
        else:
            bigplus[:, i] = funcname(devplus[i, :], param)
            bigminus[:, i] = funcname(devminus[i, :], param)
    
    bigMat = zeros((height,length))
    for i in range(0,length):
        bigMat[:,i] = (bigplus[:, i] - bigminus[:, i]) / dx[i]  #check syntax

    AA = array(bigMat[0:ny, nx:2 * nx])
    BB = array(bigMat[0:ny, 2 * nx:3 * nx])
    CC = array(bigMat[0:ny, 3 * nx + ny:3 * nx + 2 * ny])
    DD = array(bigMat[0:ny, 3 * nx + 2 * ny + nz:length])
    FF = array(bigMat[ny:ny + nx, 0:nx])
    GG = array(bigMat[ny:ny + nx, nx:2 * nx])
    HH = array(bigMat[ny:ny + nx, 2 * nx:3 * nx])
    JJ = array(bigMat[ny:ny + nx, 3 * nx:3 * nx + ny])
    KK = array(bigMat[ny:ny + nx, 3 * nx + ny:3 * nx + 2 * ny])
    LL = array(bigMat[ny:ny + nx, 3 * nx + 2 * ny:3 * nx + 2 * ny + nz])
    MM = array(bigMat[ny:ny + nx, 3 * nx + 2 * ny + nz:length])
    TT = log(1 + T0)

    if len(TT.shape)>1: #TT is matrix
        WW = array(TT[0:ny, :])
        TT = array(TT[ny:ny + nx, :])
    elif len(TT.shape)==1: #TT is vector
        WW = array(TT[0:ny]) 
        TT = array(TT[ny:ny + nx])
    else: #TT is scalar then nx=1, ny=0
        WW = zeros(0)
        TT = TT

    AA = AA if AA else zeros((ny, nx))
    BB = BB if BB else zeros((ny, nx))
    CC = CC if CC else zeros((ny, ny))
    DD = DD if DD else zeros((ny, nz))

    return [AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT]
