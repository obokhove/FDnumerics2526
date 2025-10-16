# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 12:22:42 2025

@author: wfrt0938

Solution for the steady state case as given in Question 2) a) using Euler Forward Discretization 
"""
import numpy as np

def steadystate(J): 
    """calculate steady state solution for J points"""
    #Constants defined in 2a
    Q=0.99 #constant of integration 
    alpha= 0.4709 #dimensionless parameter
    beta= 1.0 #dimensionless parameter
    ab=alpha/beta #alpha/beta, calculated here so not in loop 
    Qb= Q/beta #q/beta, calculated here so not in loop 
    H= 1 #dike height 
    bb=1.178164343 #intial value for dike width, at z=0 b = bb
    
    #Set up grid points 
    #J = 41 #number of grid points
    delta_z= H/J #stepsize for z 
    B=np.zeros(((J),1)) #array for results 
    B[0]= bb #intial condition
    
    for j in range(0,J-1): 
        B[j+1]=B[j]+(delta_z*ab)-(delta_z*Qb*(B[j]**(-3)))
    Z= np.linspace(0,1,(J)) #values for z axis
    return B,Z

"""
import matplotlib.pyplot as plt
[B,Z]=steadystate(100000)
plt.title('Height vs. Dike Width for Steady State Solution' )
halfb=B/2
plt.plot(halfb,Z, color='blue')
plt.plot((-1*halfb),Z, color='blue')
plt.xlabel('Dike Wall Position (-b/2 to + b/2)')
plt.ylabel('Height (z)')
"""
