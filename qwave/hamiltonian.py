"""
hamiltonian.py

Functions to evaluate the kinetic and potential energy functions

"""

# import modules
import numpy as np
from .utilities import *
#from scipy.interpolate import interp1d

def eval_kin(grid_points):
    T = np.zeros(grid_points**2).reshape(grid_points, grid_points)

    for i in range(grid_points):               # Create Kinetic Energy Matirix. Utilizes fourth-ordered central difference approximation
        for j in range(grid_points):
            if i == j:
                T[i,j] = -30/12
            elif abs(i - j) == 1:
                T[i,j] = 16/12
            elif abs(i-j) == 2:
                T[i,j] = -1/12
            else:
                T[i,j] = 0

    return T

def eval_pot(grid_points,grid,box_length,pot_func):
    V = np.zeros(grid_points**2).reshape(grid_points,grid_points)

    for i in range(grid_points):
        for j in range(grid_points):
            if i == j:
                V[i,j] = pes(pot_func,grid[i],box_length)
            else:
                V[i,j] = 0
    return V


