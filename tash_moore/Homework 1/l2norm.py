# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:36:12 2025

@author: wfrt0938
"""
import numpy as np
import matplotlib.pyplot as plt
from steadystate import steadystate
from nonlinearsolver import nonlinearsolver
def l2norm(J):
    #J=21
    J_ss=100001 
    H=1
    A=(J_ss-1)/(J-1) #Z[n] same value as Z_ss[An]
    Z= np.linspace(0,H,J)
    delta_z= H/(J-1) #stepsize for z 
    [B_ss,Z_ss]=steadystate(J_ss)#steady state solution, higher resolution 
    B= nonlinearsolver(J,2)
    e=np.zeros((J,1))
    for j in range(0,J):
            e[j]=B[j]-B_ss[int(j*A)]
        
    e2=np.square(e)
    e2sum=0
    for j in range(1,J):
        e2sum=e2[(j-1)]+e2[j]
    e2integral=0.5*delta_z*e2sum
    l2=(e2integral)**(0.5)
    return l2, delta_z






"""
plt.plot(Z,e2,marker='x',linestyle='-')
plt.xlabel('z')
plt.ylabel('error^2')
plt.title('J=41' )
plt.figure()
halfb_ss=B_ss/2
plt.plot(halfb_ss,Z_ss, color='blue',label='Steady state solution')
plt.plot((-1*halfb_ss),Z_ss, color='blue')
plt.title('Height vs. Dike Width for J=21' )
plt.xlabel('Dike Wall Position (-b/2 to + b/2)')
plt.ylabel('Height (z)')

halfb=B/2
plt.plot(halfb,Z, color='red',label='Time Dependant solution')
plt.plot((-1*halfb),Z, color='red')
plt.legend()
"""