#######################################################################
#            Exercise-example 1, numerics for PDE's Part II
#            within Fluid Dynamics module MATH5453 2025
#               (O. Bokhove: o.bokhove@leeds.ac.uk)
#            Solution to wave equations:
#            \eta_{tt}-\partial_x(g H(x)\partial_x \eta) = 0
#######################################################################


##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import os
import errno
import matplotlib.pyplot as plt
import time

##################################################################
# Define parameters
##################################################################
L = 6.0
grav = 9.81
H0 = 1.0
Nx = 30
Nm = 4
Linf = np.linspace(1, Nm, Nm) 
L2 = np.linspace(1, Nm, Nm) 
dxn = np.linspace(1, Nm, Nm) 
NcNx = np.linspace(1, Nm, Nm) 
print("Linf dxn",Linf,dxn)
plt.ion()
for nnx in range(1, Nm+1):
    dx = L/(1.0*Nx)
    xj = np.linspace(0.5*dx, L-0.5*dx, Nx) 
    #
    # Initial condition
    #
    m = 4
    Am = 1.0
    omeg = np.sqrt(grav*H0)*m*np.pi/L
    Tp = 2.0*np.pi/omeg
    Tend = 1.1*Tp # 10*Tp
    tijd = 0.0
    Nt = 5
    dtmeet = Tend/Nt # Tp/(1.0*Nt)
    tmeet = dtmeet # 9.0*Tp
    eta = Am*np.cos(m*np.pi*xj/L)
    etanew = eta
    etas = eta
    ppc = 0.0*eta
    pnew = ppc
    pps = 0.0*eta # -omeg*Am*np.cos(m*np.pi*xj/L)*np.sin(omeg*tijd);
    HHx = np.linspace(0.0,L, Nx+2) 
    HHx = H0+0.0*HHx;
    c0 = np.sqrt(grav*np.min(HHx));
    CFL = 0.005
    # print("0,1,Nx-1,Nx: ",HHx[0],HHx[1],HHx[Nx-1],HHx[Nx])
    # print("HHx:", HHx)
    #
    # Plot intial condition
    #
    plt.figure(11)
    plt.subplot(211)
    plt.plot(xj,eta,'-k',lw=2)
    plt.plot(xj,etas,'--r',lw=2)
    plt.xlabel('$x$',fontsize=18)
    plt.ylabel('$\eta(x,t)$',fontsize=18)
    plt.axis([0, L, -1.1*Am, 1.1*Am])
    plt.subplot(212)
    plt.plot(xj,ppc,'-k',lw=2)
    plt.plot(xj,pps,'--r',lw=2)
    plt.xlabel('$x$',fontsize=18)
    plt.ylabel('$p(x,t)$',fontsize=18)
    plt.axis([0, L, -1.1*omeg*Am, 1.1*omeg*Am]);
    plt.pause(0.5)
    #
    # Enter time loop
    #        
    dt = CFL*dx/c0
    # print("xj, eta",xj,eta,dt)
    while tijd <= Tend:
        dt = CFL*dx/c0
        tijd = tijd+dt
        #` ??? Comment out discretization:
        # First udpate ppc with forward Euler:
        pnew[0] = ppc[0]+dt*grav*( HHx[0]*(eta[1]-eta[0]) )/dx**2
        pnew[1:Nx-1] = ppc[1:Nx-1]+dt*grav*( HHx[2:Nx]*(eta[2:Nx]-eta[1:Nx-1]) - HHx[1:Nx-1]*(eta[1:Nx-1]-eta[0:Nx-2]) )/dx**2 # Note: one extra point!
        pnew[Nx-1] = ppc[Nx-1]+dt*grav*(-HHx[Nx-1]*(eta[Nx-1]-eta[Nx-2]) )/dx**2				   
        ppc = pnew				   
        # Then use this new update to update eta in backward Euler fashion:
        etanew = eta+dt*ppc
        eta = etanew
        # ??? End comment out.
        #
        # Plot solution at set times
        #
        
        if tijd>=tmeet:
            tmeet = tmeet+dtmeet
            plt.figure(11)
            plt.subplot(211)
            etas = Am*np.cos(m*np.pi*xj/L)*np.cos(omeg*tijd)
            plt.plot(xj,eta,'-k',lw=1)
            plt.plot(xj,etas,'--r',lw=1)
            plt.xlabel('$x$',fontsize=18)
            plt.ylabel('$\eta(x,t)$',fontsize=18)
            plt.axis([0, L, -1.1*Am, 1.1*Am])
            plt.subplot(212)
            pps = -omeg*Am*np.cos(m*np.pi*xj/L)*np.sin(omeg*tijd)
            plt.plot(xj,ppc,'-k',lw=1)
            plt.plot(xj,pps,'--r',lw=1)
            plt.xlabel('$x$',fontsize=18)
            plt.ylabel('$p(x,t)$',fontsize=18)
            plt.axis([0, L, -1.1*omeg*Am, 1.1*omeg*Am])
            # plt.pause(0.5)
        # End plot.  
    # ??? Comment out making of Linf and L2 errors:
    Linf[nnx-1] = max(abs(eta-etas))
    L2[nnx-1] = np.sqrt( dx*0.5*((eta[0]-etas[0])**2+(eta[Nx-1]-etas[Nx-1])**2)+dx*sum( (eta[1:Nx-1]-etas[1:Nx-1])**2) )/np.sqrt(L)
    dxn[nnx-1] = dx
    NcNx[nnx-1] = Nx
    Nx = 2*Nx
    # ??? End comment out.
    # End while loop.
    print("nnx:",nnx)

print(" ------------ END OF PROGRAM ----------")
# 
# Plotting of Linfty and L2 errors and the respective expected slopes.
# 
plt.figure(12)
plt.subplot(211)
print("dxn, Linf",dxn,Linf)
# ??? Comment out plotting of Linfty error and slope:
plt.plot(np.log10(dxn),np.log10(Linf),'x',lw=2);
print("dxn0, Linf0, NcNx",dxn,Linf,NcNx)
print("dxn0, dxN, Linf[0] Inf[Nm-1]",dxn[0],dxn[Nm-1],Linf[0],Linf[Nm-1])
plt.plot([np.log10(dxn[Nm-1]),np.log10(dxn[0])],[np.log10(Linf[Nm-1]),np.log10(Linf[Nm-1])+2.0*(np.log10(dxn[0])-np.log10(dxn[Nm-1]))],'-b');
plt.plot([-1.6,-0.6],[-3.5,-1.5],'--k') # ??? End comment out.
plt.xlabel('$ln(\Delta x)$',fontsize=18)
plt.ylabel('$ln(L^\infty)$',fontsize=18)
plt.subplot(212)
plt.plot(np.log10(dxn),np.log10(L2),'x',lw=2)
plt.plot([np.log10(dxn[Nm-1]),np.log10(dxn[0])],[np.log10(L2[Nm-1]),np.log10(L2[Nm-1])+2.0*(np.log10(dxn[0])-np.log10(dxn[Nm-1]))],'-b')
plt.plot([-1.6,-0.6],[-3.5,-1.5],'--k')
plt.xlabel('$ln(\Delta x)$',fontsize=18)
plt.ylabel('$ln(L_2)$',fontsize=18)

plt.show(block=True)
print(" ------------ END OF PROGRAM ----------")  

##################################################################    
#                        END OF PROGRAM                          #
##################################################################

