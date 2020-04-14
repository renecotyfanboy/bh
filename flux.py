# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 19:06:57 2020

@author: simd9
"""
import numpy as np
import matplotlib.pyplot as plt

x_max = 15

def flux(x):
    
    return x**(-5/2)/(x-3/2)*(x**0.5-3**0.5+np.sqrt(3/8)*np.log((np.sqrt(2)-1)/(np.sqrt(2)+1)*(np.sqrt(x)+np.sqrt(3/2))/(np.sqrt(x)-np.sqrt(3/2))))

x_span = np.linspace(3,x_max,1000)
flux_span = np.zeros_like(x_span)

for i,x in enumerate(x_span):
    
    flux_span[i] = flux(x)
    
plt.plot(x_span,flux_span)
plt.xlabel('$x$')
plt.ylabel('$\Phi (x)$')
plt.tight_layout()
plt.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
plt.xlim(3,x_max)