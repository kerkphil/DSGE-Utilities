'''
Example Using the LinApp Bundle
Linear Approximation of a simple RBC model.  This is the neoclassical
growth model of Ramesey, Cass and Koopmans with a stochastic productivity
shock added.
X(t-1) = k(t) = k
X(t) = k(t+1) = kp
X(t+1) = k(t+2) = kpp
Y(t) is empty
Z(t) = z(t) = z
Z(t+1) = z(t+1) = zp
Definitions for y(t), w(t), r(t), c(t) and i(t) are given in the 
eample_def function.
The Euler equation is given in the example_dyn function
'''

import numpy as np
import matplotlib.pyplot as plt
from LinApp_Deriv import LinApp_Deriv
from LinApp_Solve import LinApp_Solve
from LinApp_FindSS import LinApp_FindSS
 

def example_def(kp, k, z, param):
	# calculate definitions
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
	Gamma = cnow**(-theta)-cplus**(-theta)*(1+rplus-delta)/((1+g)**theta*(1+rho))
	return Gamma

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
nx = 1
ny = 0
nz = 1
Zbar = np.array([0.])

# program
nobs = 300      # number of periods in simulation
kstart = 1      # starting value for simulation (proportional to kbar)
solve = 1       # set to 1 to compute coeffsd, 0 if coeffs already in memory

if solve == 1:
    # calculate steady state values
    guessXY = np.array([1.])
    kbar = LinApp_FindSS(example_dyn,param,guessXY,Zbar,nx,ny)
    print 'kbar value is ', kbar
    zbar = Zbar
    ybar, cbar, ibar, rbar, wbar = example_def(kbar, kbar, zbar, param)
    # kbar = (((1+rho)*(1+g)**theta-1+delta)/alpha)**(1/(alpha-1))
    # ybar = kbar**alpha
    # rbar = alpha*ybar/kbar
    # wbar = (1-alpha)*ybar
    # cbar = wbar + (1+rbar-delta)*kbar - (1+g)*(1+n)*kbar
    # ibar = ybar - cbar
    reportbar = np.array([[ybar],[cbar],[ibar],[kbar],[wbar],[rbar]])
    
    # check SS values
    invec = np.concatenate([kbar, kbar, kbar, zbar, zbar])
    check = example_dyn(invec, param)
    print 'SS check value is ', check
    
    # find derivatives
    [AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, WW, TT] = \
        LinApp_Deriv(example_dyn,param,invec,1,0,1,0);
    print 'FF value is ', FF
    print 'GG value is ', GG
    print 'HH value is ', HH
    print 'LL value is ', LL
    print 'MM value is ', MM
    
    # find policy function coefficients
    PP, QQ, UU, RR, SS, VV = \
        LinApp_Solve(AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,WW,TT,phi,0,1);
    print 'PP value is ', PP
    print 'QQ value is ', QQ

# perform simulation
eps = np.random.randn(nobs)*sigma
z = np.zeros(nobs+1)
k = np.zeros(nobs+1)
y = np.zeros(nobs)
r = np.zeros(nobs)
w = np.zeros(nobs)
i = np.zeros(nobs)
c = np.zeros(nobs)
k[0] = kbar*kstart
z[0] = eps[0]
for t in range(0, nobs):
    z[t+1] = phi*z[t] + eps[t]
    k[t+1] = PP*(k[t]-kbar) + QQ*z[t] + kbar
    y[t], c[t], i[t], r[t], w[t] = example_def(k[t+1], k[t], z[t], param)
k = k[0:nobs];
z = z[0:nobs];

# plot data
t = range(0, nobs)
plt.plot(t, y, label='y')
plt.plot(t, c, label='c')
plt.plot(t, i, label='i')
plt.plot(t, k, label='k')
plt.plot(t, r, label='r')
plt.plot(t, w, label='w')
plt.plot(t, z, label='z')
plt.xlabel('time')
plt.legend(loc=9, ncol=7, bbox_to_anchor=(0., 1.02, 1., .102))
plt.show(3)