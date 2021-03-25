"""
hamiltonian.py

Functions to evaluate the kinetic and potential energy functions

"""

# import modules
import numpy as np
from scipy import constants

#import internal modules
from .utilities import *

s_cm = constants.physical_constants['speed of light in vacuum'][0] *100
B_m = constants.physical_constants['Bohr radius'][0]
j_ev = constants.physical_constants['joule-electron volt relationship'][0]
ev_h = constants.physical_constants['electron volt-hartree relationship'][0]
am_kg = constants.physical_constants['atomic mass constant'][0]
au_kg = constants.physical_constants['atomic unit of mass'][0]

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

def eval_pot(grid_points,grid,box_length,pot_func,fit_type):
    V = np.zeros(grid_points**2).reshape(grid_points,grid_points)

    for i in range(grid_points):
        for j in range(grid_points):
            if i == j:
                V[i,j] = pes(pot_func,grid[i],box_length,fit_type)
            else:
                V[i,j] = 0
    return V


def eval_pot_HO(frequency,grid_points,grid,mass):
    V = np.zeros(grid_points**2).reshape(grid_points,grid_points)
            
    for i in range(grid_points):
        for j in range(grid_points):
            if i == j:
                V[i,j] = (2*np.pi**2)*(frequency**2)*(s_cm**2)*(grid[i]**2)*((B_m)**2)*mass*(au_kg)*j_ev*ev_h

            else:
                V[i,j] = 0

    return V

