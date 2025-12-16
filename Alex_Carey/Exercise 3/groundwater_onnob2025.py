########################################
# Imports
########################################
from firedrake import *
# packages Working code as of May 14th 2018 by Will Booker and Onno Bokhove
# import numpy as np
from math import pow
import time as tijd  # OB2025: alias for timing utilities (not used below)
import numpy as np  # OB2025: numerical utilities; used for np.power in boundary term
import matplotlib   # OB2025: plotting backend (not used below)

########################################
# Mesh and Coordinates
########################################
m  = 20              # number of 1D mesh cells (elements) along y
Ly = 0.85            # domain length in y-direction
dy = Ly/m            # uniform cell size (mesh spacing) in y
mesh = IntervalMesh(m, 0 , Ly)  # 1D mesh from y=0 to y=Ly with m cells
# OB2025 y = mesh.coordinates # OB2025 Mesh coordinates
y, = SpatialCoordinate(mesh)   # continuous spatial coordinate function y on the mesh

########################################
# Time Definitions
########################################
t   = 0.0            # current simulation time
end = 150.0          # final simulation time
Ntm = 75             # number of measurement/diagnostic outputs
dtmeas = end/Ntm     # interval between outputs
tmeas = dtmeas       # next output time threshold

########################################
# Function Space
########################################
# Initially we will use a continuous linear Lagrange basis
# Try other order, 1, 2, 3
nCG = 3                              # polynomial degree for CG space
V = FunctionSpace(mesh, "CG", nCG)   # scalar continuous Galerkin space P^nCG on mesh

########################################
# Timestep
########################################
CFL = 2.3                          # stability/control factor used to set timestep
Dt = CFL*0.5*dy*dy                 # explicit-like dt estimate ~ O(dy^2); used as fixed step here
# Dt = 16*Dt
# dt.assign(CFL*0.5*dy*dy)

dt = Constant(Dt)                  # Firedrake Constant for dt used in variational forms

########################################
# Time Discretization
########################################
theta = 0.5                        # time discretization parameter: 0=explicit, 1/2=CN, 1=implicit

########################################
# Physical Constants and Case Selection
########################################
mpor  = 0.3                        # effective porosity (-)
sigma = 0.8                        # storativity/porosity scaling (-)
Lc    = 0.05                       # characteristic canal width/length scale (m)
kperm = 1e-8                       # permeability (m^2)
w     = 0.1                        # canal width (m) [not used downstream]
R     = 0.000125                   # recharge/source term (m/s)
nu    = 1.0e-6                     # kinematic viscosity of water (m^2/s)
g     = 9.81                       # gravitational acceleration (m/s^2)
alpha = kperm/( nu * mpor * sigma )  # hydraulic diffusivity coefficient in PDE
gam   = Lc/( mpor*sigma )            # canal-storage coupling coefficient
fac2  = sqrt(g)/( mpor*sigma )       # weir-law prefactor in boundary flux
# ncase = 0 Dirichlet bc, ncase = 1 overflow groundwater into canal section with weir equation:
nncase = 1                         # selects boundary condition model (0: Dirichlet, 1: weir-law)

########################################
# Initial Condition
########################################
# OB2025: h_prev = Function(V)
# OB2025 old stiff commented out: h_prev.interpolate(Expression("0.0"))
# OB2025
h_prev = Function(V).interpolate(0.0 + 0.0*y)  # initial groundwater head h^n (flat zero field)

########################################
# Output
########################################
outfile = VTKFile("./Results/groundwater_onnob.pvd")  # PVD output file for Paraview

outfile.write(h_prev , t = t )   # write initial state at t=0

########################################
# Variational Forms
########################################
# h will be the equivalent to h^n+1 in our timestepping scheme

phi = TestFunction(V)            # test function for variational formulation

def flux ( h , phi , R ):        # phi is test function q in (31) and (32)
    return ( alpha * g * h * dot ( grad (h) , grad (phi) ) - (R * phi )/ ( mpor * sigma ) )

## NB: Linear solves use TrialFunctions, non-linear solves use Functions with initial guesses.

########################################
# Boundary Conditions and Solvers

# Possible Formulations:
# nncase = 0: Dirichlet BC at y=Ly
# nncase = 1: Weir-law BC at y=0 coupling to canal section
# Possible time discretisation given Weir-law BC:
# theta = 0: explicit/IMEX time stepping
# theta = 0.5: Crank-Nicholson time stepping
# theta = 1: implicit time stepping
########################################
if nncase == 0:
    # Provide intial guess to non linear solve
     h = Function(V)                                       # unknown groundwater head at new time
     h.assign(h_prev)                                      # initial guess for nonlinear iteration
     F = ( (h-h_prev)*phi/dt  + theta * flux ( h , phi , R ) + (1-theta)* flux ( h_prev, phi, R) ) *dx  # residual form
    # Boundary conditions: Condition at Ly satisfied weakly
     bc1 = DirichletBC(V, 0.07, 1)                        # Dirichlet head at boundary id 1 (y=Ly)
     h_problem = NonlinearVariationalProblem( F , h , bcs = bc1)  # nonlinear variational problem

elif nncase == 1:
   if theta == 0.0: # Matches (31)
      h, out = TrialFunction(V), Function(V)              # trial (unknown) and solution storage for linear step
      aa = (h*phi/dt)*dx+(gam*phi*h/dt)*ds(1)             # left-hand side bilinear form with canal coupling
      L2 = ( h_prev*phi/dt - flux ( h_prev, phi, R) ) *dx # interior right-hand side using previous step
      L = L2+( gam*phi*h_prev/dt-phi*fac2*max_value(2.0*h_prev/3.0,0.0)*sqrt(max_value(2.0*h_prev/3.0,0.0)) )*ds(1) # Matches (29): boundary RHS with weir law
      explicit_problem = LinearVariationalProblem(aa, L, out)  # linear problem for explicit theta=0
      explicit_solver = LinearVariationalSolver(explicit_problem, solver_parameters={'mat_type':'aij', 
          'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps','ksp_rtol': 1e-14})  # direct solve
   elif theta > 0.0: # Matches (30) when theta=1/2
      h = Function(V)                                     # unknown groundwater head at new time
      h.assign(h_prev)                                    # start Newton iteration from previous head
      F = ( (h-h_prev)*phi/dt  + theta * flux ( h , phi , R ) + (1-theta)* flux ( h_prev, phi, R) ) *dx  # interior residual
      # Add boundary contributions at y = 0: 
      F2 = ( gam*phi*(h-h_prev)/dt
                + theta*phi*fac2*np.power(max_value(2.0*h/3.0,0.0),1.5)
                + (1-theta)*phi*fac2*max_value(2.0*h_prev/3.0,0.0)*sqrt(max_value(2.0*h_prev/3.0,0.0)) )*ds(1)  # weir-law boundary flux
      h_problem = NonlinearVariationalProblem( F+F2 , h )         # nonlinear problem with boundary term
      h_solver = NonlinearVariationalSolver(h_problem, solver_parameters={'mat_type':'aij','ksp_type':'preonly','pc_type':'lu','pc_factor_mat_solver_type': 'mumps','ksp_rtol': 1e-14})  # direct Newton linearization solve

########################################
# Time Loop
########################################
while (t < end):
    # First we increase time
    t += Dt                                  # advance time by fixed step
    # Use the solver and then update values for next timestep
    if theta == 0.0:
        explicit_solver.solve()            # solve linear explicit/IMEX step
        h_prev.assign(out)                # update stored head with new solution
    elif theta > 0.0:
        h_solver.solve()                  # solve nonlinear step (e.g., CN/implicit)
        h_prev.assign(h)                  # update stored head with new solution
    # Write output to file for paraview visualisation
    if t>tmeas:
     print('Time is: ',t)                 # periodic progress output
     tmeas = tmeas+dtmeas                 # schedule next output time
     outfile.write(h_prev , t = t )       # write field to Paraview file
# End while time loop


























































































