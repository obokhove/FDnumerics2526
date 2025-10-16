# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:03:36 2025

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
b_T = 0.58331626 #b^0 and b_J value


#defining the grid space:
J = 4001 #number of spatial steps
dz = H/(J-1)
z = np.linspace(0,H,J) #using z1 so this can be used for later steady state plotting
C = np.zeros((J,1)) #using C for steady state solution so it can be used for later plots/calculations

C[0] = b_B #defining the initial value of b

for j in range (1,J):
    C[j] = C[j-1] + (a*dz)/b - (Q*dz)/(b*((C[j-1])**3))

##Question 2 Part c:


def stride(Nx,Nx1):
  stride = (Nx - 1) // (Nx1 - 1)

  return stride

#number of grid points considered
J1 = 11
J2 = 21
J3 = 41
J4 = 81
J5 = 161

#steady state at chosen number of grid points - to match up with the numerical solution for error calculation
S1 = C[::stride(J,J1)]
S2 = C[::stride(J,J2)]
S3 = C[::stride(J,J3)]
S4 = C[::stride(J,J4)]
S5 = C[::stride(J,J5)]


#time grid points
dt = 0.00001
T = 2 
M = int(T/dt)


##first pt calculation

#defining useful quantities:
dz = (H/(J1-1))
mu = dt/((dz)**2)
A = 3*a*mu*dz
K = (b*mu)/2

#defining matrix for solution and calculating at each time step
B = np.zeros((J1,M))
B[:,0] = b_T #initial condition
B[0,0] = b_B #lower boundary condition

for n in range (0,M-1):
     for k in range (1,J1-1):
         B[k,n+1] = (B[k,n] - A*((B[k,n])**2)*(B[k,n]-B[k-1,n]) 
                     + K*(((B[k,n])**3)*(B[k+1,n] - 2*B[k,n] + B[k-1,n]) + ((B[k+1,n])**3)*(B[k+1,n] - B[k,n])
                          - ((B[k-1,n])**3)*(B[k,n] - B[k-1,n])))
     B[0,n+1] = b_B #lower boundary condition
     B[J1-1,n+1] = b_T #upper boundary condition
       
#redefining final time step column as the numerical solution      
B_sol1 = np.zeros((J1,1))
for j in range (0,J1):
    B_sol1[j] = B[j,M-1]
    
e1 = np.zeros((J1,1)) #defining vector for error

#calculating error vector using previously defined exact solution and numerical solution
for j in range (0,J1):
    e1[j] = S1[j] - B_sol1[j]
    
    
## second grid no. calculation - following the same process as the first 

dz = (H/(J2-1))
mu = dt/((dz)**2)
A = 3*a*mu*dz
K = (b*mu)/2
    
B = np.zeros((J2,M))
B[:,0] = b_T
B[0,0] = b_B

for n in range (0,M-1):
     for k in range (1,J2-1):
         B[k,n+1] = (B[k,n] - A*((B[k,n])**2)*(B[k,n]-B[k-1,n]) 
                     + K*(((B[k,n])**3)*(B[k+1,n] - 2*B[k,n] + B[k-1,n]) + ((B[k+1,n])**3)*(B[k+1,n] - B[k,n])
                          - ((B[k-1,n])**3)*(B[k,n] - B[k-1,n])))
     B[0,n+1] = b_B
     B[J2-1,n+1] = b_T
       
       
B_sol2 = np.zeros((J2,1))
for j in range (0,J2):
    B_sol2[j] = B[j,M-1]
    
e2 = np.zeros((J2,1)) #defining vector for error

for j in range (0,J2):
    e2[j] = S2[j] - B_sol2[j]


## third grid no. calculation

dz = (H/(J3-1))
mu = dt/((dz)**2)
A = 3*a*mu*dz
K = (b*mu)/2
    
B = np.zeros((J3,M))
B[:,0] = b_T
B[0,0] = b_B

for n in range (0,M-1):
     for k in range (1,J3-1):
         B[k,n+1] = (B[k,n] - A*((B[k,n])**2)*(B[k,n]-B[k-1,n]) 
                     + K*(((B[k,n])**3)*(B[k+1,n] - 2*B[k,n] + B[k-1,n]) + ((B[k+1,n])**3)*(B[k+1,n] - B[k,n])
                          - ((B[k-1,n])**3)*(B[k,n] - B[k-1,n])))
     B[0,n+1] = b_B
     B[J3-1,n+1] = b_T
       
       
B_sol3 = np.zeros((J3,1))
for j in range (0,J3):
    B_sol3[j] = B[j,M-1]
    
e3 = np.zeros((J3,1)) #defining vector for error

for j in range (0,J3):
    e3[j] = S3[j] - B_sol3[j]


## fourth grid no. calculation

dz = (H/(J4-1))
mu = dt/((dz)**2)
A = 3*a*mu*dz
K = (b*mu)/2
    
B = np.zeros((J4,M))
B[:,0] = b_T
B[0,0] = b_B

for n in range (0,M-1):
     for k in range (1,J4-1):
         B[k,n+1] = (B[k,n] - A*((B[k,n])**2)*(B[k,n]-B[k-1,n]) 
                     + K*(((B[k,n])**3)*(B[k+1,n] - 2*B[k,n] + B[k-1,n]) + ((B[k+1,n])**3)*(B[k+1,n] - B[k,n])
                          - ((B[k-1,n])**3)*(B[k,n] - B[k-1,n])))
     B[0,n+1] = b_B
     B[J4-1,n+1] = b_T
       
       
B_sol4 = np.zeros((J4,1))
for j in range (0,J4):
    B_sol4[j] = B[j,M-1]
    
e4 = np.zeros((J4,1)) #defining vector for error

for j in range (0,J4):
    e4[j] = S4[j] - B_sol4[j]
    
    
    
## fifth grid no. calculation

dz = (H/(J5-1))
mu = dt/((dz)**2)
A = 3*a*mu*dz
K = (b*mu)/2
    
B = np.zeros((J5,M))
B[:,0] = b_T
B[0,0] = b_B

for n in range (0,M-1):
     for k in range (1,J5-1):
         B[k,n+1] = (B[k,n] - A*((B[k,n])**2)*(B[k,n]-B[k-1,n]) 
                     + K*(((B[k,n])**3)*(B[k+1,n] - 2*B[k,n] + B[k-1,n]) + ((B[k+1,n])**3)*(B[k+1,n] - B[k,n])
                          - ((B[k-1,n])**3)*(B[k,n] - B[k-1,n])))
     B[0,n+1] = b_B
     B[J5-1,n+1] = b_T
       
       
B_sol5 = np.zeros((J5,1))
for j in range (0,J5):
    B_sol5[j] = B[j,M-1]
    
e5 = np.zeros((J5,1)) #defining vector for error

for j in range (0,J5):
    e5[j] = S5[j] - B_sol5[j]

 
# calculating the L_2 norm for each number of grid pts:

L_21 = e1[0]**2
for j in range (1,J1-1):
    L_21 = (L_21 + 2*((e1[j])**2))
    
L_21 = (L_21 + ((e1[J1-1])**2))
L_21 = np.sqrt((dz/2)*(L_21))


L_22 = e2[0]**2
for j in range (1,J2-1):
    L_22 = (L_22 + 2*((e2[j])**2))
    
L_22 = (L_22 + ((e2[J2-1])**2))
L_22 = np.sqrt((dz/2)*(L_22))


L_23 = e3[0]**2
for j in range (1,J3-1):
    L_23 = (L_23 + 2*((e3[j])**2))
    
L_23 = (L_23 + ((e3[J3-1])**2))
L_23 = np.sqrt((dz/2)*(L_23))


L_24 = e4[0]**2
for j in range (1,J4-1):
    L_24 = (L_24 + 2*((e4[j])**2))
    
L_24 = (L_24 + ((e4[J4-1])**2))
L_24 = np.sqrt((dz/2)*(L_24))


L_25 = e5[0]**2
for j in range (1,J5-1):
    L_25 = (L_25 + 2*((e5[j])**2))
    
L_25 = (L_25 + ((e5[J5-1])**2))
L_25 = np.sqrt((dz/2)*(L_25))


#plotting the pts using a log scaling to look at the dependence of L_2 norm compared to dz
fig5 = plt.figure()
plt.scatter(H/(J1-1),L_21)
plt.scatter(H/(J2-1),L_22)
plt.scatter(H/(J3-1),L_23)
plt.scatter(H/(J4-1),L_24)
plt.scatter(H/(J5-1),L_25)
plt.xscale("log")
plt.yscale("log")
plt.xlabel('log(dz)')
plt.ylabel('L_2 norm of error (log)')
plt.title('L_2 norm of error as a function of dz')



#Calculating the L_inf norm for each number of grid pts:  

L_inf1 = np.zeros((J1,1))

L_inf1 = abs(e1[0])

for j in range (0,J1):
    L_inf1 = max(L_inf1, abs(e1[j]))


L_inf2 = np.zeros((J2,1))

L_inf2 = abs(e2[0])

for j in range (0,J2):
    L_inf2 = max(L_inf2, abs(e2[j]))

    
L_inf3 = np.zeros((J3,1))

L_inf3 = abs(e3[0])

for j in range (0,J3):
    L_inf3 = max(L_inf3, abs(e3[j]))


L_inf4 = np.zeros((J4,1))

L_inf4 = abs(e4[0])

for j in range (0,J4):
    L_inf4 = max(L_inf4, abs(e4[j]))


L_inf5 = np.zeros((J5,1))

L_inf5 = abs(e5[0])

for j in range (0,J5):
    L_inf5 = max(L_inf5, abs(e5[j]))


#plotting the pts using a log scaling to look at the dependence of L_inf norm compared to dz
fig6 = plt.figure() 
plt.scatter(H/(J1-1),L_inf1)
plt.scatter(H/(J2-1),L_inf2)
plt.scatter(H/(J3-1),L_inf3)
plt.scatter(H/(J4-1),L_inf4)
plt.scatter(H/(J5-1),L_inf5)
plt.xscale("log")
plt.yscale("log")
plt.xlabel('log(dz)')
plt.ylabel('L_inf norm of error (log)')
plt.title('L_inf norm of error as a function of dz')
 