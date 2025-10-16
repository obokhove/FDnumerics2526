# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 10:36:43 2025

@author: qvbp0176
"""

import numpy as np #import numpy for calculations
import matplotlib.pyplot as plt #import matplotlib for plotting results

#first define the constants as given:
Q = 0.99
a = 0.4709 #alpha value
b = 1.0 #beta value
H = 1 
b_B = 1.178164343 #b_0 value
b_T = 0.585373798 #b^0 and b_J value


#defining the grid space:
J=3001 #number of spatial steps
dz = H/(J-1)
z1 = np.linspace(0,H,J) #using z1 so this can be used for later steady state plotting
C = np.zeros((J,1)) #using C for steady state solution so it can be used for later plots/calculations

C[0] = b_B #defining the initial value of b

for j in range (1,J):
    C[j] = C[j-1] + (a*dz)/b - (Q*dz)/(b*((C[j-1])**3))

fig1 = plt.figure()
plt.plot(C/2,z1,'red')
plt.plot(-C/2,z1,'red')
plt.xlabel('dike width')
plt.ylabel('z')
plt.title('Steady State Solution')


dt = 0.0001 #time step that is stable for all dz


T = [0.05, 0.1, 0.2, 0.5, 1, 2] #max value of time (T=0.05,0.1,0.2,0.5,1,2)
colours = ['blue', 'purple', 'yellow', 'orange', 'green', 'lightblue'] #for plotting


##J=11

#defining the grid space:
J = 11 #number of spatial steps (use J=11,21,41 as prescribed in question)
dz = H/(J-1)
z = np.linspace(0,H,J)

#defining useful quantities:
mu = dt/((dz)**2)
A = 3*a*mu*dz
K = (b*mu)/2

fig2 = plt.figure()
for i in range (0,6):
    M = int(T[i]/dt)  
    B = np.zeros((J,M)) #using initial condition
    B[:,0] = b_T #redefining first value as given BC at z=0
    B[0,0] = b_B
    for n in range (0,M-1):
         for k in range (1,J-1):
             B[k,n+1] = (B[k,n] - A*((B[k,n])**2)*(B[k,n]-B[k-1,n]) 
                         + K*(((B[k,n])**3)*(B[k+1,n] - 2*B[k,n] + B[k-1,n]) + ((B[k+1,n])**3)*(B[k+1,n] - B[k,n])
                              - ((B[k-1,n])**3)*(B[k,n] - B[k-1,n])))
         B[0,n+1] = b_B
         B[J-1,n+1] = b_T
           
           
    B_sol = np.zeros((J,1))
    for j in range (0,J):
        B_sol[j] = B[j,M-1]
            
    plt.plot(B_sol/2,z,colours[i], label=T[i])
    plt.plot(-B_sol/2,z,colours[i])
plt.plot(C/2,z1,'red', label='steady state')
plt.plot(-C/2,z1,'red')
plt.xlabel('dike width')
plt.ylabel('z')
plt.title('Solution for J=11')
plt.legend(title='time',loc='upper left')


##J=21

#defining the grid space:
J = 21 #number of spatial steps (use J=11,21,41 as prescribed in question)
dz = H/(J-1)
z = np.linspace(0,H,J)

#defining useful quantities:
mu = dt/((dz)**2)
A = 3*a*mu*dz
K = (b*mu)/2


fig3 = plt.figure()
for i in range (0,6):
    M = int(T[i]/dt)  
    B = np.zeros((J,M)) #using initial condition
    B[:,0] = b_T #redefining first value as given BC at z=0
    B[0,0] = b_B
    for n in range (0,M-1):
         for k in range (1,J-1):
             B[k,n+1] = (B[k,n] - A*((B[k,n])**2)*(B[k,n]-B[k-1,n]) 
                         + K*(((B[k,n])**3)*(B[k+1,n] - 2*B[k,n] + B[k-1,n]) + ((B[k+1,n])**3)*(B[k+1,n] - B[k,n])
                              - ((B[k-1,n])**3)*(B[k,n] - B[k-1,n])))
         B[0,n+1] = b_B
         B[J-1,n+1] = b_T
           
           
    B_sol = np.zeros((J,1))
    for j in range (0,J):
        B_sol[j] = B[j,M-1]
            
    plt.plot(B_sol/2,z,colours[i],label=T[i])
    plt.plot(-B_sol/2,z,colours[i])
plt.plot(C/2,z1,'red', label='steady state')
plt.plot(-C/2,z1,'red')
plt.xlabel('dike width')
plt.ylabel('z')
plt.title('Solution for J=21') 
plt.legend(title='time',loc='upper left')
    
    
##J=41 

#defining the grid space:
J = 41 #number of spatial steps (use J=11,21,41 as prescribed in question)
dz = H/(J-1)
z = np.linspace(0,H,J)

#defining useful quantities:
mu = dt/((dz)**2)
A = 3*a*mu*dz
K = (b*mu)/2


fig4 = plt.figure()
for i in range (0,6):
    M = int(T[i]/dt)  
    B = np.zeros((J,M)) #using initial condition
    B[:,0] = b_T #redefining first value as given BC at z=0
    B[0,0] = b_B
    for n in range (0,M-1):
         for k in range (1,J-1):
             B[k,n+1] = (B[k,n] - A*((B[k,n])**2)*(B[k,n]-B[k-1,n]) 
                         + K*(((B[k,n])**3)*(B[k+1,n] - 2*B[k,n] + B[k-1,n]) + ((B[k+1,n])**3)*(B[k+1,n] - B[k,n])
                              - ((B[k-1,n])**3)*(B[k,n] - B[k-1,n])))
         B[0,n+1] = b_B
         B[J-1,n+1] = b_T
           
           
    B_sol = np.zeros((J,1))
    for j in range (0,J):
        B_sol[j] = B[j,M-1]
            
    plt.plot(B_sol/2,z,colours[i],label=T[i])
    plt.plot(-B_sol/2,z,colours[i])
plt.plot(C/2,z1,'red', label='steady state')
plt.plot(-C/2,z1,'red')
plt.xlabel('dike width')
plt.ylabel('z')
plt.title('Solution for J=41')
plt.legend(title='time',loc='upper left')

