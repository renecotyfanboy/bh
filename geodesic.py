# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:44:30 2019

@author: simd9
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from ode import ode87,ode_rk4

def f(t,Z):
    
    Z_dot = np.zeros_like(Z)
    
    Z_dot[0] = Z[1]
    Z_dot[1] = 3/2*Z[0]**2-Z[0]
    
    return Z_dot

def criterion(Z):
    
    return Z[0]>=0 and Z[1]<1e20

#%% Paramètres
    
r_s = 1

#%% 

plt.figure()

u_c_span = np.hstack((np.linspace(0.5,0.99,5),1.01))

for u_c in u_c_span:

    Y0 = np.array([0,u_c*2/(3*3**0.5)])
    
    theta,Y = ode87(f,0,10*np.pi,Y0,criterion,minimal_nb_of_points=10)
    
    index_polar = Y[:,0]>0.1
    r = 1/Y[index_polar,0]
    theta_polar = theta[index_polar]
    plt.polar(theta_polar,r,label='$u_c$ = {:2.3f}'.format(u_c))

plt.legend()

#%% Portai
plt.figure()

u_c = 1.5

Y0 = np.array([0,u_c*2/(3*3**0.5)])
theta,Y = ode87(f,0,10*np.pi,Y0,criterion,minimal_nb_of_points=1000)
    
index_phase = Y[:,0]>0
theta_ph = theta[index_phase]
Y_ph = Y[index_phase,:]   
plt.plot(Y[index_phase,0],Y[index_phase,1])
plt.xlabel(r'$u$')
plt.ylabel(r"$u'$")