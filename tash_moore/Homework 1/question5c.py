# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:14:45 2025

@author: wfrt0938
"""

import numpy as np
import matplotlib.pyplot as plt
from l2norm import l2norm
[L2_11,delta_z11]=l2norm(11)
[L2_21,delta_z21]=l2norm(21)
[L2_31,delta_z31]=l2norm(31)
[L2_41,delta_z41]=l2norm(41)
[L2_51,delta_z51]=l2norm(51)
[L2_61,delta_z61]=l2norm(61)
[L2_71,delta_z71]=l2norm(71)
[L2_81,delta_z81]=l2norm(81)

L2=[L2_11,L2_21,L2_31,L2_41,L2_51,L2_61,L2_71,L2_81]
delta_z=[delta_z11,delta_z21,delta_z31,delta_z41,delta_z51,delta_z61,delta_z71,delta_z81]
plt.plot(delta_z,L2,marker='x',linestyle='-')
plt.title('L^2 Norm vs. Delta_z' )
plt.xlabel('Delta_Z')
plt.ylabel('Height (z)')