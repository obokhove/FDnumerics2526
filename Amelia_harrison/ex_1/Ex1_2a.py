# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 12:32:11 2025

@author: qlcq0853

numerics ex1 
"""
import numpy as np
import matplotlib.pyplot as plt

def steady_state():
    
    #no of grid points
    n = 1000000
    
    #value of constants
    Q=0.99
    H=1
    alpha=0.4709
    beta = 1.0
    b_b = 1.178164343
    delta_z = H/(n-1)
    
    b_sol = np.zeros(n)
    b_sol[0] = b_b
    
    #iterating through b
    for i in range(1,n):
        b_sol[i] = b_sol[i-1] + (delta_z/beta)*(alpha - Q/(b_sol[i-1])**3)
        
    #plotting b and z
    
    z_s = np.linspace(0,1,len(b_sol))
    """
    plt.plot(b_sol/2, z_s, color='red')
    plt.plot(-b_sol/2, z_s, color='red')
    plt.xlim(-b_b/1.5, b_b/1.5)
    plt.ylim(0, H)
    plt.xlabel('Dike width')
    plt.ylabel('z')
    plt.title('Steady state solution')
    print(b_sol[n-1])"""
    b_steady = b_sol
    print(b_steady)
    return b_steady, z_s
print(steady_state())