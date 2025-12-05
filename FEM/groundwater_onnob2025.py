from firedrake import *
# 
# packages Working code as of May 14th 2018 by Will Booker and Onno Bokhove
# import numpy as np
from math import pow
import time as tijd # OB2025
import numpy as np # OB2025
import matplotlib # OB2025
m  = 20
Ly = 0.85
dy = Ly/m
mesh = IntervalMesh(m, 0 , Ly)
# OB2025 y = mesh.coordinates # OB2025 Mesh coordinates
y, = SpatialCoordinate(mesh) # OB2025 

# Time definitions
t   = 0.0
end = 150.0
Ntm = 75
dtmeas = end/Ntm
tmeas = dtmeas

# 
# Define Function space on our mesh.
# Initially we will use a continuous linear Lagrange basis
# Try other order, 1, 2, 3
nCG = 3 # OB2025
V = FunctionSpace(mesh, "CG", nCG) # OB2025

# Define timestep value
CFL = 2.3
Dt = CFL*0.5*dy*dy  # Based on FD estimate; note that dt must be defined before flux, etc
# Dt = 16*Dt
# dt.assign(CFL*0.5*dy*dy)

dt = Constant(Dt) # Using dt.assign in the while loop should avoid having to rebuild the solver iirc

# Define Crank Nicholson parameter
theta = 0.5

# Define Groundwater constants
mpor  = 0.3
sigma = 0.8
Lc    = 0.05
kperm = 1e-8
w     = 0.1
R     = 0.000125
nu    = 1.0e-6
g     = 9.81
alpha = kperm/( nu * mpor * sigma )
gam   = Lc/( mpor*sigma )
fac2  = sqrt(g)/( mpor*sigma )
# 
# ncase = 0 Dirichlet bc, ncase = 1 overflow groundwater into canal section with weir equation:
nncase = 1

# Initial condition
# OB2025: h_prev = Function(V)
# OB2025 old stiff commented out: h_prev.interpolate(Expression("0.0"))
# OB2025
h_prev = Function(V).interpolate(0.0 + 0.0*y) # OB2025 IC, I guess hnum = 0.0*y h_prev.interpolate(hnum)

# Create storage for paraview
outfile = VTKFile("./Results/groundwater_onnob.pvd")

# Write IC to file for paraview
outfile.write(h_prev , t = t )

# Define trial and test functions on this function space
# h will be the equivalent to h^n+1 in our timestepping scheme

phi = TestFunction(V)

def flux ( h , phi , R ):  # phi is test function q in (31) and (32)
    return ( alpha * g * h * dot ( grad (h) , grad (phi) ) - (R * phi )/ ( mpor * sigma ) )

## NB: Linear solves use TrialFunctions, non-linear solves use Functions with initial guesses.

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
    if theta == 0.0:
           explicit_solver.solve()
           h_prev.assign(out) # has to be renamed via out
    elif theta > 0.0:
           h_solver.solve()
           h_prev.assign(h)
        # Write output to file for paraview visualisation
    if t>tmeas:
        print('Time is: ',t)
        tmeas = tmeas+dtmeas
        outfile.write(h_prev , t = t )
# End while time loop


























































































