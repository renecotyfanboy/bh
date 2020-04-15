# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:07:24 2020

@author: simd9
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from ode import ode87
from adaptive import Learner2D,BlockingRunner

class bh_imager:
    
    x_lim = (-10,10)
    y_lim = (-10,10)
    r_lim = (3,15)
    i = 5 #degree
    i *= np.pi/180
    
    def r_in_bounds(self,r):
        
        return r > self.r_lim[0] and r < self.r_lim[1]
    
    def theta_d_p(self,alpha):
        
        return np.arccos(-np.sin(alpha)*np.cos(self.i)/np.sqrt(1-np.cos(alpha)**2*np.cos(self.i)**2))
    
    def theta_d_s(self,alpha):
        
        return np.arccos(-np.sin(alpha)*np.cos(self.i)/np.sqrt(1-np.cos(alpha)**2*np.cos(self.i)**2))+np.pi
    
    @staticmethod
    def f(t,Z):
        
        Z_dot = np.zeros_like(Z)
        Z_dot[0] = Z[1]
        Z_dot[1] = 3/2*Z[0]**2-Z[0]
        
        return Z_dot
    
    @staticmethod
    def criterion(Z):
        
        return Z[0]>=0 and Z[1]<1e20
    
    def calc_r_p(self,alpha,b):

        u_c = 3*3**0.5/2/b
        Y0 = np.array([0,u_c*2/(3*3**0.5)])
        theta_f = self.theta_d_p(alpha)
        theta,Y = ode87(self.f,0,theta_f,Y0,self.criterion,minimal_nb_of_points=10)
        
        return 1/Y[-1,0]
    
    def calc_r_s(self,alpha,b):
        
        u_c = 3*3**0.5/2/b
        Y0 = np.array([0,u_c*2/(3*3**0.5)])
        theta_f = self.theta_d_s(alpha)
        theta,Y = ode87(self.f,0,theta_f,Y0,self.criterion,minimal_nb_of_points=10)
        
        return 1/Y[-1,0]
    
    def find_r(self,xy):
        
        x,y = xy
        alpha = np.angle(x+1j*y)
        b = np.abs(x+1j*y)
        
        r_p = self.calc_r_p(alpha,b)
        
        if self.r_in_bounds(r_p):
            
            return r_p
        
        r_s = self.calc_r_s(alpha,b)
        
        if self.r_in_bounds(r_s):
            
            return r_s
        
        return 0
        
    def learner(self):
        
        learner = Learner2D(self.find_r, bounds=[self.x_lim, self.y_lim])
        runner = BlockingRunner(learner, goal=lambda l: l.loss() < 0.1)
        
        return learner
#%%

imager = bh_imager()
learner = imager.learner()

def plot(learner):
    plot = learner.plot(tri_alpha=0.2)
    return (plot.Image + plot.EdgePaths.I + plot).cols(2)

#plot(learner)