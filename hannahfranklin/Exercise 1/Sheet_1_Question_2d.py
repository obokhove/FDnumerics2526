# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:33:48 2025

@author: qvbp0176
"""

import numpy as np #import numpy for calculations
import matplotlib.pyplot as plt #import matplotlib for plotting results


z_ref = 0.9 #km
a = 0.4709 #alpha
H = 3 #km

J = 1001 #will consider J = 11, 21, 41
z = np.linspace(0,H,J)
dz = H/(J-1)

T = [0.1, 0.5, 1, 2, 3, 4, 5]
M = 10001
colours = ['blue', 'purple', 'yellow', 'orange', 'green', 'red', 'lightblue']


b = np.zeros((J,M))
for i in range (0,J):
    b[i,:] = i * (1/(J-1))
    
z_val = np.zeros((J,M)) #calculates value of z at each time step for given b between [0,1]

#note: use b to calculate the corresponding z 
fig1 = plt.figure()
for j in range (0,7):
    dt = T[j]/(M-1)
    for n in range (0,M):
        t = n*dt
        for i in range (0,J):
            z_val[i,n] = (z_ref + a*t + (1/a)*(b[i,n] - np.arctanh(b[i,n])))
    z_exact = np.zeros((J,1))
    for i in range (0,J):
        z_exact[i] = max(z_val[i,M-1],0)
        z_exact[i] = min(z_exact[i],3)
        
    plt.plot(b/2,z_exact,colours[j])
    plt.plot(-b/2,z_exact,colours[j])
    
plt.xlabel('dike width (m)')
plt.ylabel('z (km)')
plt.title('Exact Solution and J=11 approximation')
#plt.show()


#b_B and b_T for each considered time
b_B = [0.8642,0.8908,0.9158,0.9486,0.9680,0.9798,0.9872]
b_T = [0,0,0,0,0,0,0.6422]


J = 11 #run over J =11,21,41
dz = 3/(J-1)
z = np.linspace(0,H,J)

for i in range (0,7):
    dt = T[i]/(M-1)
    B = np.zeros((J,M)) #using initial condition
    bT = b_T[i]
    bB = b_B[i]
    for n in range (0,M-1):
        # b_T = min(z_val[:,n]) #defining variable b_T
        # b_T = max(b_T, 0)
        # b_B = max(z_val[:,n]) #defining variable b_B
        # b_B = min(b_B,3)
        mu = dt/((dz)**2)
        A = 3*a*mu*dz
        K = (mu)/2
        B[:,0] = bT #redefining first value as given BC at z=0
        B[0,0] = bB
        for k in range (1,J-1):
             B[k,n+1] = (B[k,n] - A*((B[k,n])**2)*(B[k,n]-B[k-1,n]) 
                         + K*(((B[k,n])**3)*(B[k+1,n] - 2*B[k,n] + B[k-1,n]) + ((B[k+1,n])**3)*(B[k+1,n] - B[k,n])
                              - ((B[k-1,n])**3)*(B[k,n] - B[k-1,n])))
        B[0,n+1] = bB
        B[J-1,n+1] = bT
           
           
    B_sol = np.zeros((J,1))
    for j in range (0,J):
        B_sol[j] = B[j,M-1]
            
    plt.plot(B_sol/2,z, colours[i],label=T[i], linestyle='--' )
    plt.plot(-B_sol/2,z, colours[i], linestyle='--' )
    plt.legend(title='time',loc='upper left')
 