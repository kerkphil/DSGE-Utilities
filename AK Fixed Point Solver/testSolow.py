# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 13:31:47 2017

@author: Kerk Phillips
"""

from AKsolve import AKsolve
    
def example(X, fparams):
    # An example using the Solow growth model
    Xnew = (1-delta)*X + gamma*X**alpha
    
    return Xnew
   
# set parameter values for AKsolve     
ccrit = 1e-12   # convegence criterion for A-K solution
conv = .25      # initital weight on new values in convexifier
maxiter = 10000 # maximum number of iterations allowed in A-K solver
shrinkon = 1    # binary flag to implement shrinking conv value
shrink = .75    # factor to reduce conv if A-K not converging
expandon = 0    # binary flag to implement expanding conv value
expand = 1.01   # factor to expand conv if A-K is converging
disttype = 1    # distance measure formula
display = 0     # display iterations on or off

# set parameter values for example
delta = .08
gamma = .05
alpha = .33
fparams = {delta, gamma, alpha}

# Use AK method to find fixed point
Xfixed, distance, count = AKsolve(1.0, example, fparams, ccrit, conv, \
    maxiter, shrinkon, shrink, expandon, expand, disttype, display)
    
# find closed form solution for SS value
Xbar = (gamma/delta)**(1/(1-alpha))

print('Xfixed:    ', Xfixed)
print('Xbar       ', Xbar)
print('distance:  ', distance)
print('iterations:', count)