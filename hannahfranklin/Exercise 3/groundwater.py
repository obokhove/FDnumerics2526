from firedrake import *
from math import pow
import time as tijd # OB2025
import numpy as np # OB2025
import matplotlib.pyplot as plt # OB2025
import random
import pandas as pd

#Spatial mesh:
m  = 10 #number of spatial steps - 10,20,40,80
Ly = 0.85 #length of channel considered
dy = Ly/m
mesh = IntervalMesh(m, 0 , Ly)
# OB2025 y = mesh.coordinates # OB2025 Mesh coordinates
y, = SpatialCoordinate(mesh) # OB2025
yvals = np.linspace(0,Ly,m) #values for y at each spatial step

#Time definitions:
t   = 0.0 #starting time
end = 100 #end time
Nt = end/2
#dtmeas = end/Ntm
#tmeas = dtmeas

#Defining reporting intervals for hm and hcm
dtmeas_hm = 10 #for measuring and storing hm every 10s
dtmeas_hcm = 2 #for measuring and storing hcm and R(t) every 2s
meas_hm = dtmeas_hm
meas_hcm = dtmeas_hcm
meas_R = meas_hcm

#Defining and choosing timestep value Dt:
CFL = 2.3
Dt = CFL*0.5*dy*dy  # Based on FD estimate; note that dt must be defined before flux, etc - to ensure scheme is stable
# Dt * (1/9) for CG3 removes oscillations - scheme is stable again
dt = Constant(Dt) #Firedrake constant form of Dt
print('dt =',float(dt))
# Dt = 16*Dt
# dt.assign(CFL*0.5*dy*dy)

#Defining Function space:
nCG = 1 #try orders: 1,2,3
V = FunctionSpace(mesh, "CG", nCG) # OB2025

#Defining Crank Nicholson parameter
theta = 0 # 0 for explicit scheme, 0.5 for Crank-Nicholson scheme

#Defining Groundwater constants:
mpor  = 0.3 #porosity
sigma = 0.8 #the fraction of the pore that can be filled
Lc    = 0.05 #length of canal
kperm = 1e-8 #permeability
w     = 0.1 #average width of channel
R     = Constant(0.000125) #max rainfall
Rmax  = 0.000125
dur   = 10 #duration of constant rainfall
nu    = 1.0e-6 #kinematic viscosity
g     = 9.81 #acceleration due to gravity
#Useful coefficients:
alpha = kperm/( nu * mpor * sigma )
gam   = Lc/( mpor*sigma )
fac2  = sqrt(g)/( mpor*sigma )

#Case selection:
#ncase = 0 Dirichlet bc, constant water level in canal
#ncase = 1 overflow groundwater into canal section with weir equation
nncase = 1

#Choice if rainfall is constant or variable:
R_var = 1 #0 if constant, 1 if variable

if R_var == 0:
  rainfall = 'constant'
elif R_var == 1:
  rainfall = 'variable'

R_options = [1,2,4,9] #options for durations of variable rainfall
#Creating a function for randomly varying rainfall:
intnow = -1
def Rval(t):
  global R_var, R_options, dur, intnow #variables defined globally
  Rmax = 0.000125
  if R_var == 0:
    dur = 10
    return Rmax #constant rainfall value
  elif R_var == 1:
    interval = int(t/10) #period number - intervals of 10s
    if interval != intnow:
      intnow = interval
      dur = random.choice(R_options) #randomly choosing a duration of rainfall over the time period
    if t < (dur + 10*interval): #time period of duration of rainfall, using randomly chosen period
      return Rmax #rainfall occurs at rate of constant rainfall initially
    else:
      return 0 #no rainfall occurs outside of the randomly chosen time period

R.assign(Rval(t))
print('Rainfall duration is', dur,'s')

#Initial conditions:
h_prev = Function(V).interpolate(0.0 + 0.0*y) #ground water level is initially 0
hcm = 0 #inital level of canal water

#Creating output for paraview
outfile = VTKFile("./Results/groundwater_onnob.pvd")
outfile.write(h_prev , t = t )

# Define trial and test functions on this function space
# h will be the equivalent to h^n+1 in our timestepping scheme

#Creating a test function:
phi = TestFunction(V)

#Creating a function for the flux:
def flux ( h , phi , R ):  # phi is test function q in (31) and (32)
    return ( alpha * g * h * dot ( grad (h) , grad (phi) ) - (R * phi )/ ( mpor * sigma ) )

## NB: Linear solves use TrialFunctions, non-linear solves use Functions with initial guesses.

#Plotting of initial values:
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.set_ylabel(r'$h_{m}(y,t)$ (m)')
ax1.set_xlabel(r'$y$ (m)')
ax1.grid()
ax2.set_ylabel(r'$h_{cm}(y,t)$ (m)')
ax2.set_xlabel(r'$t$ (s)')
ax2.set_xlim(0,100)
ax2.grid()
ax3.set_ylabel(r'$R(t)$ (m)')
ax3.set_xlabel(r'$t$ (s)')
ax3.set_xlim(0,100)
ax3.grid()

#Creating storage arrays for values calculated:
t_arr = np.array([0])
hm_arr = np.array([h_prev.at(y) for y in yvals])
hcm_arr = np.array([h_prev.at(0)])
R_arr = np.array(Rmax)
time_arr = np.array([0])

#Plotting initial profile hm:
ax1.plot(yvals, hm_arr, label = 't=0s')

if nncase == 0:
   # Provide intial guess to non linear solve
   h = Function(V)
   h.assign(h_prev)
   F = ( (h-h_prev)*phi/dt  + theta * flux ( h , phi , R ) + (1-theta)* flux ( h_prev, phi, R) ) *dx
   # Boundary conditions: Condition at Ly satisfied weakly
   bc1 = DirichletBC(V, 0.07, 1)
   h_problem = NonlinearVariationalProblem( F , h , bcs = bc1)

elif nncase == 1:
   if theta == 0.0: # Matches (31)
     h, out = TrialFunction(V), Function(V) # Has to be set for linear solver
     aa = (h*phi/dt)*dx+(gam*phi*h/dt)*ds(1)
     L2 = ( h_prev*phi/dt - flux ( h_prev, phi, R) ) *dx
     L = L2+( gam*phi*h_prev/dt-phi*fac2*max_value(2.0*h_prev/3.0,0.0)*sqrt(max_value(2.0*h_prev/3.0,0.0)) )*ds(1) # Matches (29)
     explicit_problem = LinearVariationalProblem(aa, L, out)
     explicit_solver = LinearVariationalSolver(explicit_problem, solver_parameters={'mat_type':'aij',
        'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps','ksp_rtol': 1e-14})
   elif theta > 0.0: # Matches (30) when theta=1/2
     h = Function(V)
     h.assign(h_prev)
     F = ( (h-h_prev)*phi/dt  + theta * flux ( h , phi , R ) + (1-theta)* flux ( h_prev, phi, R) ) *dx
     # Add boundary contributions at y = 0:
     F2 = ( gam*phi*(h-h_prev)/dt+theta*phi*fac2*np.power(max_value(2.0*h/3.0,0.0),1.5)+(1-theta)*phi*fac2*max_value(2.0*h_prev/3.0,0.0)*sqrt(max_value(2.0*h_prev/3.0,0.0)) )*ds(1)
     h_problem = NonlinearVariationalProblem( F+F2 , h )
     h_solver = NonlinearVariationalSolver(h_problem, solver_parameters={'mat_type':'aij','ksp_type':'preonly','pc_type':'lu','pc_factor_mat_solver_type': 'mumps','ksp_rtol': 1e-14})

# Time loop
while (t < end):
    # First we increase time
    t += Dt
    # Print to console the current time
    # Use the solver and then update values for next timestep
    t_arr = np.append(t_arr,t)
    R.assign(Rval(t))
    R_arr = np.append(R_arr,Rval(t))
    if theta == 0.0: #explicit scheme
           explicit_solver.solve()
           h_prev.assign(out) # has to be renamed via out
    elif theta > 0.0: #for use when evaluating the Crank-Nicholson scheme
           h_solver.solve()
           h_prev.assign(h)
        # Write output to file for paraview visualisation
    if t>meas_hcm: #outputting hcm every 2s
        print('Time is: ',t)
        time_arr = np.append(time_arr, t)
        meas_hcm = meas_hcm + dtmeas_hcm
        outfile.write(h_prev , t = t )
        hcm_arr = np.append(hcm_arr, [h_prev.at(0)])
        #R_arr = np.append(R_arr,R)
    if t>meas_hm: #outputting hm every 10s
      print('Rainfall duration is', dur,'s') #printing duration of rainfall in 10s period.
      meas_hm = meas_hm + dtmeas_hm
      hm_arr = np.array([h_prev.at(y) for y in yvals])
      ax1.plot(yvals, hm_arr, label = 't='+str(t)+'s') #plotting the profile in the channel every 10s

# End while time loop
ax1.plot(yvals, hm_arr, '--k', label = 't=100s')
ax2.plot(time_arr, hcm_arr, '--k')
ax3.plot(t_arr, R_arr, '--k')
ax1.legend(title='time (s)', bbox_to_anchor = (1.5,1))
ax1.set_title('dy=Ly/'+ str(m) +', CG='+ str(nCG) +', rainfall '+ str(rainfall))
plt.show()
df = pd.DataFrame({"hm(y,t)": hm_arr, "y": yvals})
df.to_csv(f"GW_Ny={m}_CG={nCG}_rainfall={rainfall}.csv") #exporting data for comparison to other meshes

#print('Final value of $h_{cm}$ is:', hcm_arr[int(Nt)]) #printing steady state value of hcm
