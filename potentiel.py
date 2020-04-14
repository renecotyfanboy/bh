# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:58:21 2020

@author: simd9
"""

import numpy as np
import matplotlib.collections as collections
import matplotlib.pyplot as plt

def V(u,b):
    
    return -(4/27)*(1/b)**2 + (u**2)*(1-u)

u = np.linspace(0,1,1000)
b_span = [0.90,0.95,1.00,1.10,1.25,1.50,2.00]

fig, ax = plt.subplots()

#collection = collections.BrokenBarHCollection.span_where(
#    u, ymin=0, ymax=0.25, where=np.ones_like(u)>0, facecolor='green', alpha=0.5)
#ax.add_collection(collection)
#
#collection = collections.BrokenBarHCollection.span_where(
#    u, ymin=-0.25, ymax=0, where=np.ones_like(u)>0, facecolor='red', alpha=0.5)
#ax.add_collection(collection)

for b in b_span:

    ax.plot(u,V(u,b),label=r'$b = % 6.2f b_c$'%b)
    
ax.axhline(y=0,color='black',linestyle='-.')
    
ax.grid()
plt.xlabel(r'$u$')
plt.ylabel(r'$V(u)$')
plt.legend()
plt.tight_layout()