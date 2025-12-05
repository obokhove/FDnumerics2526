## FEM exercise and notes
OB: Groundwater code and exercise Part-II added on 05-12-2025.
OB: Slides updated on 05-112-2025.

Groundwater:
Firedrake produces results in the form of .vtu files which are grouped together
in a .pvd file. We visualise this data using paraview. Select the green apply button. Now you can visualise the data either changing the colour scheme, which is recommended 2d or 3d data, or for the 1d
examples we can use warpbyscalar. Go to filters/alphabetical/warpbyscalar. Now click the (green) apply button again. The time bar can now advance the simulation through time. 

In 1D, got to filters/alphabetical/plotoverline On left panel choose “Bot tom axis fixed” and choose a maximum. In view/animationview set the speed of the animation, etc. One can snap through by hand to see the profiles one by one. (OB 05-12-205: This worked in older Paraview but not yet in Paraview 5.13 --please report when you got it to work in 5.13 or higher, and how.)

## Firedrake

Load Firedrake from https://www.firedrakeproject.org/documentation.html either the recommended native version or Docker version.

## Finite-element exercises (check/adapt instructions)

The exercise on FEM modelling via Firedrake is found in the FEM folder.

Sample Firedrake programs:
- for Poisson equation (codes in named folder) via weak formulation and minimisation (Yang Lu with Robin Furze, added comments by OB)
- Load Firedrake environment, go to directory with (downloaded) file
- To run code type >> python3 Poissons_eq_v2.py

## Paraview instructions

:new: *Warning* (via TH from IT): When other modules have been loaded incompatible libraries then Paraview may not work.
E.g., the "anaconda3" module will cause Qt problems with "paraview". Clean environment by running:

`module purge`

`module load paraview-rhel8/5.11.2`

`paraview`

One should not add any "module load" commands to their .bashrc files to load
modules automatically as this can cause problems with standard system software and
with other modules.  one may need to fix .bashrc files if the purge command
doesn't clear the anaconda3 libraries.

## Paraview visualisation
Load output file and display:
- Open Paraview
- Go to directory with the output file named "output.pvd"
- Under "File" in menu click open and find that file "output.pvd" (click okay).
- Choose u1 or u2 and then click "Apply"
