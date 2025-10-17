# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:05:04 2025
Solution for nonlinear convection-diffusion equation (5) with variable timestep implemented 

@author: wfrt0938
"""

import numpy as np
def nonlinearsolver(J,t_end): 
    """steadycalculate solution for J points and end time t_end"""

    #Constants defined in Question 2
    alpha= 0.4709 #dimensionless parameter
    beta= 1.0 #dimensionless parameter  
    H= 1 #dike height 
    bb=1.178164343 #boundary condition value for dike width, b(0,t)=bb
    bt=0.585373798#boundary and intial condition value for dike width, b(H,t)=bt and b(z,0)=bt
    #t_end=0.05 #run to this time 
    
    #Set up grid points 
    #J =41 #number of grid points in z
    delta_z= H/(J-1) #stepsize for z 
    #T=np.zeros((50000,2)) #array to monitor timesteps out of interest, not neccessary for code to work 
    Mu=np.zeros((J-2,1)) #array for Mu
    B=np.zeros((J,1)) #array for B
    B_new=np.zeros((J,1)) #array B, to be updated at each time step 
    B[:,0]= bt #set intial condition
    B[0,:]=bb #set boundary condition 
    B[(J-1),:]=bt #set boundary condition
    t=0
    #count=0
    while t < t_end:
        B_new[0]=bb #boundary conditions
        B_new[J-1]=bt #boundary condition 
        #Find the maximum value of mu and delta_t that will be stable for all values of j 
        for i in range(1,(J-1)): #find the maximum value of mu for each value of j, from 5.e)
            mu1=6*alpha*delta_z*(B[i]**2) 
            mu2=2*(B[i]**3)+(B[i-1]**3)+(B[i+1]**3)
            mu=2/(mu1+(beta*mu2))
            Mu[i-1]=mu
        mu=min(Mu) #want the smallest value, in order to ensure stability for all j 
        delta_t=mu*(delta_z**2)
        for j in range(1,(J-1)):
            #update value of b 
            term2=3*mu*alpha*delta_z*(B[j]**2)*(B[j]-B[j-1]) #split equation in order to aid debugging 
            term3=0.5*beta*mu*((B[j]**3)+(B[j+1]**3))*(B[j+1]-B[j])
            term4=0.5*beta*mu*((B[j]**3)+(B[j-1]**3))*(B[j]-B[j-1])
            B_new[j]=B[j]-term2+term3-term4
        B=B_new
        t=t+delta_t #update time
        #T[count,0]=t[0] #monitor timesteps out of interest, not neccessary for code to work 
        #T[count,1]=delta_t[0]
        #count=count+1
    return B

"""
import matplotlib.pyplot as plt
B=nonlinearsolver(100,0.05)
Z= np.linspace(0,1,100)
halfb=B/2
plt.plot(halfb,Z, color='blue')
plt.plot((-1*halfb),Z, color='blue')
plt.xlabel('Dike Wall Position (-b/2 to + b/2)')
plt.ylabel('Height (z)')

"""