## Finite-volume or discontinuous Galerkin leading-order or Godunov method (FV or DG0)

Update in (5c; typo corrected) and Exercise.

Codes for TC0 and TC1 uploaded; please check and test.

NB: TC2 was incomplete. 16-11: TC2a: cst influx; and TC2b: time-dependent influx now work;
There were a lot of typos in the (plotting) expressions. Also hb and wb have been made into hbb and wbb as fd.Constant(-).

Firedrake (installation and examples)
https://www.firedrakeproject.org/documentation.html

Two download options: follow linux instructions and use chatgpt or use the "dockers image" (link part way donw the documentation page).

The first few questions will be theoretical questions on the formulation of the numerical flux and Godunov scheme.

The implementation of the finite-volume scheme as DG0 finite element method into Firedrake is explain in the Appendix of the exercise, which should be studied in detail. In addition, consider the Firedrake example: https://www.firedrakeproject.org/demos/DG_advection.py.html

# Running Firedrake from docker:
Start Docker engine in Applications folder, then in a termianl:
docker run -it \
  -v /Users/onnobokhove/amtob/werk/vuurdraak2021/blexact:/home/firedrake/workdir \
  -w /home/firedrake/workdir \
  firedrakeproject/firedrake:latest

Replace "/Users/onnobokhove/amtob/werk/vuurdraak2021/blexact:" by your directory, i.e. where your program is.

For running under Windows, please asl CDT student Jonny Bolton.

Also from CDT student Robin Furze:
"If they are using the university system then we use Apptainer which doesn't require Docker desktop, but IT service desk may have to install that for them. A quicker solution if they aren't doing anything too intensive (from https://fem-on-colab.github.io/packages.html):

Go to https://colab.research.google.com/
File>Upload notebook>Browse>FiredrakeColab.ipynb (attached)

This will temporarily install Firedrake on an online notebook which doesn't require anything extra from IT or any installation on the computer itself. The caveat is that the installation isn't cached so there is a ~2 min wait every new session, and it won't be appropriate for anything too memory intensive.
Happy to pop over if anyone needs help with this."
