# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 14:53:53 2020

@author: simd9
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

x_max = 7

def redshift(x,alpha,b,inclinaison):
    
    inclinaison *= np.pi/180
    
    if(x>3 and x < x_max):
    
        return 1/np.sqrt(1-3/(2*x))*(1+np.cos(inclinaison)*np.cos(alpha)*(3/(2*x))**(3/2)*b/(3*3**0.5/2))
    
    else :
        
        return 1

for inclinaison in [5*i for i in range(1,19)]:
    
    x_span = np.linspace(-10,10,1000)
    y_span = np.linspace(-10,10,1000)
    X,Y = np.meshgrid(x_span,y_span)
    alpha = np.angle(X+1j*Y)
    b = np.abs(X+1j*Y)
    
    R_p = np.loadtxt("image/rp_{}.txt".format(inclinaison))
    R_s = np.loadtxt("image/rs_{}.txt".format(inclinaison))
    
    R_p[R_p>15] = 0
    R_s[R_s>15] = 0
    
    redshift_matrix = np.zeros((1000,1000))
    
    for i,x in enumerate(x_span):
        for j,y in enumerate(y_span):
            
            redshift_matrix[i,j] = redshift(R_p[i,j],alpha[i,j],b[i,j],inclinaison) - 1
            
            if(redshift_matrix[i,j] == 0):
                
                redshift_matrix[i,j] =redshift(R_s[i,j],alpha[i,j],b[i,j],inclinaison) - 1
            
    fig, ax = plt.subplots(figsize=(5,5))


    im = ax.imshow(redshift_matrix, 
                    interpolation='bilinear',
                    cmap=cm.RdBu_r,
                    norm=colors.Normalize(vmin=-1, vmax=1),
                    aspect='auto',
                    origin='lower',
                    extent=[-7,7,-7,7])
    
    cbar = fig.colorbar(im,ax=ax)
    
    plt.tight_layout()
    #plt.savefig("image_final/rshift_{}.png".format(inclinaison))
    plt.close()