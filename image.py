# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 14:53:53 2020

@author: simd9
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors

x_max = 15

def redshift(x,alpha,b,inclinaison):
    
    inclinaison *= np.pi/180
    
    if(x>3):
    
        return 1/np.sqrt(1-3/(2*x))*(1+np.cos(inclinaison)*np.cos(alpha)*(3/(2*x))**(3/2)*b/(3*3**0.5/2))
    
    else :
        
        return 1

    
def flux(x):
    if(x>3) and (x<x_max):
        
        return x**(-5/2)/(x-3/2)*(x**0.5-3**0.5+np.sqrt(3/8)*np.log((np.sqrt(2)-1)/(np.sqrt(2)+1)*(np.sqrt(x)+np.sqrt(3/2))/(np.sqrt(x)-np.sqrt(3/2))))
    else:
        return 0
    
for inclinaison in [5*i for i in range(1,19)]:
    
    print(inclinaison)
    
    x_span = np.linspace(-10,10,1000)
    y_span = np.linspace(-10,10,1000)
    X,Y = np.meshgrid(x_span,y_span)
    alpha = np.angle(X+1j*Y)
    b = np.abs(X+1j*Y)
    
    R_p = np.loadtxt("image/rp_{}.txt".format(inclinaison))
    R_s = np.loadtxt("image/rs_{}.txt".format(inclinaison))
    
    image = np.zeros((1000,1000))
    
    for i,x in enumerate(x_span):
        for j,y in enumerate(y_span):
            #**(-4)
            image[i,j] = flux(R_p[i,j])/(redshift(R_p[i,j],alpha[i,j],b[i,j],inclinaison)**4)
            
            if image[i,j] == 0:
                #(redshift(R_s[i,j],alpha[i,j],b[i,j],inclinaison))**(-4)
                image[i,j] = flux(R_s[i,j])/(redshift(R_s[i,j],alpha[i,j],b[i,j],inclinaison)**4)
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(-7,7)
    ax.set_ylim(-7,7)
    im = ax.imshow(image, 
                    interpolation='bilinear',
                    cmap=cm.hot,
                    norm=colors.Normalize(vmin=image.min(), vmax=image.max()),
                    aspect='auto',
                    origin='lower',
                    extent=[-7,7,-7,7])
    plt.axis('off')
    plt.tight_layout()
    #plt.savefig("image_final/img_{}.png".format(inclinaison))
    plt.close()