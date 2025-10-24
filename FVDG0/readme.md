## Finite-volume or discontinuous Galerkin leading-order or Godunov method (FV or DG0)

Firedrake (installation and examples)
https://www.firedrakeproject.org/documentation.html

Two download options: follow linux instructions and use chatgpt or use the "dockers image" (link part way donw the documentation page).

The first few questions will be theoretical questions on the formulation of the numerical flux and Godunov scheme.

# Running Firedrake from docker:
Start Docker engine in Applications folder, then in a termianl:
docker run -it \
  -v /Users/onnobokhove/amtob/werk/vuurdraak2021/blexact:/home/firedrake/workdir \
  -w /home/firedrake/workdir \
  firedrakeproject/firedrake:latest

Replace "/Users/onnobokhove/amtob/werk/vuurdraak2021/blexact:" by your directory, i.e. where your program is.

For running under Windows, please asl CDT student Jonny Bolton.
