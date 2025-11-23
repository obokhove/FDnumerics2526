##Error analysis for numerics sheet 2 test cases:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import os

current_dir = os.path.dirname(__file__)#

csv_path = os.path.join(current_dir, "TC0_fine_Nx=10000_CFL=0.125.csv")
df = pd.read_csv(csv_path,
                 skiprows=0, engine = "python",)

csv_path1 = os.path.join(current_dir, "TC0_4dx_Nx=1250_CFL=0.125.csv")
df1 = pd.read_csv(csv_path1,
                 skiprows=0, engine = "python",)

csv_path2 = os.path.join(current_dir, "TC0_2dx_Nx=2500_CFL=0.25.csv")
df2 = pd.read_csv(csv_path2,
                 skiprows=0, engine = "python",)

csv_path3 = os.path.join(current_dir, "TC0_base_Nx=5000_CFL=0.5.csv")
df3 = pd.read_csv(csv_path3,
                 skiprows=0, engine = "python",)

csv_path4 = os.path.join(current_dir, "TC0_0.5dx_Nx=10000_CFL=1.0.csv")
df4 = pd.read_csv(csv_path4,
                 skiprows=0, engine = "python",)


A1 = df1["A(s,t)"].to_numpy()
A2 = df2["A(s,t)"].to_numpy()
A3 = df3["A(s,t)"].to_numpy()
A4 = df4["A(s,t)"].to_numpy()
A = df["A(s,t)"].to_numpy()

s1 = df1["s"].to_numpy()
s2 = df2["s"].to_numpy()
s3 = df3["s"].to_numpy()
s4 = df4["s"].to_numpy()
s = df["s"].to_numpy()


a1 = np.zeros((1,2500))
for i in range (0,2500):
     a1[0,i] = A[8*i]
     
a2 = np.zeros((1,5000))  
for i in range (0,5000):
    a2[0,i] = A[4*i]
    
a3 = np.zeros((1,10000))
for i in range (0,10000):
    a3[0,i] = A[2*i]
 
    
a1 = np.asarray(a1).ravel()
a2 = np.asarray(a2).ravel()
a3 = np.asarray(a3).ravel()

e1 = A1 - a1
e2 = A2 - a2
e3 = A3 - a3
e4 = A4 - A 

J = 100
N1 = int(2500/J)
N2 = int(5000/J)
N3 = int(10000/J)
N4 = int(20000/J)
S1 = np.zeros((J,1))
S2 = np.zeros((J,1))
S3 = np.zeros((J,1))
S4 = np.zeros((J,1))
E1 = np.zeros((J,1))
E2 = np.zeros((J,1))
E3 = np.zeros((J,1))
E4 = np.zeros((J,1))
for i in range (0,J):
    S1[i] = s1[N1*i]
    S2[i] = s2[N2*i]
    S3[i] = s3[N3*i]
    S4[i] = s4[N4*i]
    E1[i] = e1[N1*i]
    E2[i] = e2[N2*i]
    E3[i] = e3[N3*i]
    E4[i] = e4[N4*i]


plt.figure(1)
plt.plot(S1,E1,label = "4dx")
plt.plot(S2,E2,label = "2dx")
plt.plot(S3,E3,label = "dx")
plt.plot(S4,E4,label = "0.5dx")
plt.xlabel("s (m)")
plt.ylabel("error (m^2)")
plt.title("TC0 error in numerical approximation for area (varying dx)")
plt.legend()
plt.show()



#calculating the L2 norm:

Lx = 5000 #length of river considered
#numbers of x grid points:
Nx1 = 1250
Nx2 = 2500
Nx3 = 5000
Nx4 = 10000

#dx values for each considered number of gridpoints:
dx1 = Lx/Nx1
dx2 = Lx/Nx2
dx3 = Lx/Nx3
dx4 = Lx/Nx4
 
#L2 norm for dx1   
L_21 = e1[0]**2
for j in range (1,J*N1-1):
    L_21 = (L_21 + 2*((e1[j])**2))
    
L_21 = (L_21 + ((e1[(J*N1)-1])**2))
L_21 = np.sqrt((dx1/2)*(L_21))

#L2 norm for dx2   
L_22 = e2[0]**2
for j in range (1,J*N2-1):
    L_22 = (L_22 + 2*((e2[j])**2))
    
L_22 = (L_22 + ((e2[(J*N2)-1])**2))
L_22 = np.sqrt((dx2/2)*(L_22))

#L2 norm for dx3   
L_23 = e3[0]**2
for j in range (1,J*N3-1):
    L_23 = (L_23 + 2*((e3[j])**2))
    
L_23 = (L_23 + ((e3[(J*N3)-1])**2))
L_23 = np.sqrt((dx3/2)*(L_23))

#L2 norm for dx4   
L_24 = e4[0]**2
for j in range (1,J*N4-1):
    L_24 = (L_24 + 2*((e4[j])**2))
    
L_24 = (L_24 + ((e4[(J*N4)-1])**2))
L_24 = np.sqrt((dx4/2)*(L_24))

#for line connecting pts:
x1 = np.linspace(dx1,dx4,100)
x2 = np.linspace(L_21,L_24,100)


plt.figure(2)
plt.scatter(dx1,L_21, label="4dx")
plt.scatter(dx2,L_22, label="2dx")
plt.scatter(dx3,L_23, label="dx")
plt.scatter(dx4,L_24,label="0.5dx")
plt.plot(x1,x2,'--b')

plt.xlabel('dx')
plt.ylabel('L_2 norm of error')
plt.title('TC0 L_2 norm of error as a function of dx')
plt.legend()
