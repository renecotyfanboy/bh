# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:55:46 2019

@author: simd9
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#%%
    
for inclinaison in [5*i for i in range(1,19)]:

    x = np.linspace(-10,10,1000)
    y = np.linspace(-10,10,1000)
    R_p = np.loadtxt("image/rp_{}.txt".format(inclinaison))
    R_s = np.loadtxt("image/rs_{}.txt".format(inclinaison))
    
    levels = [3+0.25*i for i in range(10)]
    fig, ax = plt.subplots(figsize=(5,5))
    CS_p = ax.contour(x , y, R_p,levels=levels,cmap=cm.inferno)
    CS_s = ax.contour(x , y, R_s,levels=levels,cmap=cm.inferno)
    
    for i in range(len(levels)):
        CS_p.collections[i].set_label(levels[i])
    
    plt.legend(loc='upper left')
    ax.legend()
    ax.set_xlim(-7,7)
    ax.set_ylim(-7,7)
    plt.tight_layout()
    
    #plt.savefig("image_final/isor_{}.png".format(inclinaison))
    #plt.close()

