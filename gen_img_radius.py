# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:55:46 2019

@author: simd9
"""

if __name__ == '__main__':
    
    from dask.distributed import Client,LocalCluster
    import numpy as np
    import dask
    from ode import ode87
    
    cluster = LocalCluster()
    client = Client(cluster)
    
    def f(t,Z):
        
        Z_dot = np.zeros_like(Z)
        
        Z_dot[0] = Z[1]
        Z_dot[1] = 3/2*Z[0]**2-Z[0]
        
        return Z_dot
    
    def criterion(Z):
        
        return Z[0]>=0 and Z[1]<1e20
    
    def theta_d_p(alpha,i):
        
        return np.arccos(-np.sin(alpha)*np.cos(i)/np.sqrt(1-np.cos(alpha)**2*np.cos(i)**2))
    
    def theta_d_s(alpha,i):
        
        return np.arccos(-np.sin(alpha)*np.cos(i)/np.sqrt(1-np.cos(alpha)**2*np.cos(i)**2)) +np.pi
    
    def calc_u_p(alpha,b,inclinaison):
        
        u_c = 3*3**0.5/2/b
        Y0 = np.array([0,u_c*2/(3*3**0.5)])
        theta_f = theta_d_p(alpha,inclinaison)
        theta,Y = ode87(f,0,theta_f,Y0,criterion,minimal_nb_of_points=10)
        
        return 1/Y[-1,0]
    
    def calc_u_s(alpha,b,inclinaison):
        
        u_c = 3*3**0.5/2/b
        Y0 = np.array([0,u_c*2/(3*3**0.5)])
        theta_f = theta_d_s(alpha,inclinaison)
        theta,Y = ode87(f,0,theta_f,Y0,criterion,minimal_nb_of_points=10)
        
        return 1/Y[-1,0]
    
    def compute_radius(alpha,b,inclinaison):
            
        shape = np.shape(alpha)
        
        R_p = []
        R_s = []
        
        for i in range(shape[0]):
            R_p.append([])
            R_s.append([])
            for j in range(shape[1]):
                R_p[-1].append(dask.delayed(calc_u_p)(alpha[i,j],b[i,j],inclinaison))
                R_s[-1].append(dask.delayed(calc_u_s)(alpha[i,j],b[i,j],inclinaison))
        
        R_p = dask.compute(*R_p)
        R_s = dask.compute(*R_s)
        
        R_p = np.hstack((np.flip(R_p,axis = 1),R_p))
        R_s = np.hstack((np.flip(R_s,axis = 1),R_s))
        
        return R_p,R_s
    
    #%% Propriétés de l'image
    
    n_points = 1000
    x_span = np.linspace(0,10,n_points//2)
    y_span = np.linspace(-10,10,n_points)
    X,Y = np.meshgrid(x_span,y_span)
    
    inclinaisons = [5*i for i in range(1,19)]
        
    for inclinaison in inclinaisons:
    
        i = inclinaison*np.pi/180
        
        alpha = np.angle(X+1j*Y)
        b = np.abs(X+1j*Y)
        
        R_p,R_s = compute_radius(alpha,b,i)
        
        np.savetxt('/scratch/students/s.dupourque/image/rp_{}.txt'.format(inclinaison),R_p)
        np.savetxt('/scratch/students/s.dupourque/image/rs_{}.txt'.format(inclinaison),R_s)
