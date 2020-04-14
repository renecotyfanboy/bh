# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:55:46 2019

@author: simd9
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#%%

def theta_d_p(alpha,i):
    
    i *= np.pi/180
    
    return np.arccos(-np.sin(alpha)*np.cos(i)/np.sqrt(1-np.cos(alpha)**2*np.cos(i)**2))

for inclinaison in [5*i for i in range(1,19)]:

    x_span = np.linspace(-10,10,1000)
    y_span = np.linspace(-10,10,1000)
    X,Y = np.meshgrid(x_span,y_span)
    alpha = np.angle(X+1j*Y)
    b = np.abs(X+1j*Y)

    R_p = np.zeros_like(b)
    
    for i,x in enumerate(x_span):
        for j,y in enumerate(y_span):
            
            R_p[i,j] = b[i,j]/np.sin(theta_d_p(alpha[i,j],inclinaison))
    
    levels = [3+0.25*i for i in range(10)]
    fig, ax = plt.subplots(figsize=(5,5))
    CS_p = ax.contour(x_span , y_span, R_p,levels=levels,cmap=cm.inferno)
    
    for i in range(len(levels)):
        CS_p.collections[i].set_label(levels[i])
    
    plt.legend(loc='upper left')
    ax.legend()
    ax.set_xlim(-7,7)
    ax.set_ylim(-7,7)
    plt.tight_layout()
    
    #plt.savefig("image_final/isor_newton_{}.png".format(inclinaison))
    plt.close()