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

def calculate_kinetic(grid_points: int) -> np.ndarray:
    """
    Calculate the Kinetic Energy Matirix.
    Utilizes fourth-ordered central difference approximation.
    Parameters:
        grid_points: int
    Returns:
        T np.ndarray
    """
    T = np.zeros((grid_points, grid_points))

    for i in range(grid_points):
        T[i,i] = -30/12
        for j in range(grid_points):
            if abs(i - j) == 1:
                T[i,j] = 16/12
            elif abs(i-j) == 2:
                T[i,j] = -1/12
            else:
                pass

    return T

def calculate_potential(grid_points: int, grid, box_length, potential_func, fit_type) -> np.ndarray:
    """
    Calculate the Potential Energy Matirix.
    Parameters:
        grid_points: int
        grid: np.ndarray
        box_length: float
        potential_func: str
        fit_type: str
    Returns:
        V: np.ndarray          
    """
    V = np.zeros((grid_points, grid_points))

    for i in range(grid_points):
        V[i,i] = pes(potential_func, grid[i], fit_type)
    return V


def calculate_HO_potential(frequency, grid_points, grid, mass):
    V = np.zeros((grid_points, grid_points))
            
    for i in range(grid_points):
        V[i,i] = (2*np.pi**2)*(frequency**2)*(s_cm**2)*(grid[i]**2)*\
                    ((B_m)**2)*mass*(au_kg)*j_ev*ev_h

    return V
