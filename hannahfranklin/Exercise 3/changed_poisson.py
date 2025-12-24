from firedrake import *
import matplotlib.pyplot as plt
import numpy as np

#EXERCISE 6 - DIFFERENT FUNCTION AND BCS
nx = ny = 16 # Try various mesh resolutions, starting coarse, say 16x16 etc.
# nx = 16, 32, 64, 128, 256, 512
# h = 1/nx - mesh resolution for different {h,p} pairs

mesh = UnitSquareMesh(nx,ny,quadrilateral=True)

#Function space generation:
V = FunctionSpace(mesh, 'CG', 1) # Piecewise linear continuous Galerkin function space or polynomials
#FunctionSpace(mesh, 'CG', p) - change p=1,2,3,4 to consider different {h,p} pairs

#
# Method 1: construct the weak form manually by multiplying and manipulating the Poisson equation and solve the linear system
#
#STEP 1:
u = TrialFunction(V) # The unknown or variable u(x,y)
v = TestFunction(V)  # The testfunction of u, which may be better called delu or deltau

x, y = SpatialCoordinate(mesh) # Mesh coordinates

f = Function(V).interpolate((5/4)*pi**2*sin((pi/2)*x)*cos(pi*y)) # chosen function f(x,y)

#STEP 1:
#weak form for Poisson equation - no use of FEM expansion u~u_1 in this formulation, just using unknown solution u and test function v.
a = (inner(grad(u),grad(v)))*dx #weak form first term
L = (f*v)*dx #weak form second term

#STEP 2:
#FEM expansion using u~u_1
u_1 = Function(V, name='u_1') # Name of solution for first method

#imposing the dirichlet boundary conditions at x=0,1
bc_x0 = DirichletBC(V, Constant(0), 1) # Dirichlet boundary conditions imposed at x=0
bc_x1 = DirichletBC(V, cos(pi *y), 2) # BC at x=1, a function of y
# The homogeneous Neumann boundary conditions are "automatically" included, i.e. do not need anything explicit

#STEP 4:
#solving the algebraic system to find u_1
solve(a == L, u_1, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}, bcs=[bc_x0,bc_x1]) # Step 4: the solution assigned to u1


#
# Method 2:
#
#Ritz-Galerkin principle, finding solution u_2 that minimises the functional J[u] =Ju

u_2 = Function(V, name='u_2') # Name of solution for second method

#STEP 2:
#discretisation of Ritz-Galerkin principle using the FEM expansion u~u_2

#Defining the functional J[u]
Ju = (0.5*inner(grad(u_2),grad(u_2)) - u_2*f)*dx # f->ULF? Step 2

#derivative of functional in direction of variation v
F = derivative(Ju, u_2, du=v) # Step 2/3: The weak form generated

#not needed for firedrake, but this is where STEP 3 would be implemented to introduce a local coordinate system and reference coordinates.

#STEP 4:
#solving F=0 to find u_2 - function that minimises functional J[u].
solve(F == 0, u_2, bcs=[bc_x0, bc_x1]) # Step 4: the solution assigned to u2


#contour plotting to visualise the solution:

#creating a meshgrid of points in the domain
x = np.linspace(0,1,nx+1)
y = np.linspace(0,1,ny+1)
X,Y = np.meshgrid(x,y)

#matrix of values for u_1 and u_2 at each point on the grid:
U1 = np.zeros((nx+1,ny+1))
U2 = np.zeros((nx+1,ny+1))
for i in range (nx+1):
  for j in range (ny+1):
    U1[i,j] = u_1.at([X[i,j],Y[i,j]])
    U2[i,j] = u_2.at([X[i,j],Y[i,j]])

Uexact = np.sin((np.pi/2) * X)*np.cos(np.pi * Y) #exact solution

Udiff = abs(U2 - Uexact)

#plotting contours of u_2 solution, and the difference between the solutions:

#u_2 contour:
plt.figure(1)
contour2 = plt.contourf(X,Y,U2, levels=20)
plt.title('Numerical Solution obtained through method 2 (u_2)')
plt.colorbar(contour2)
plt.xlabel('x')
plt.ylabel('y')

#error between u_2 solution and exact solution contour:
plt.figure(2)
contour3 = plt.contourf(X,Y,Udiff, levels=20)
plt.title('Difference between numerical solution & exact solution')
plt.colorbar(contour3, pad=0.2)
plt.xlabel('x')
plt.ylabel('y')
