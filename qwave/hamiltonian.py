"""
hamiltonian.py

Functions to evaluate the kinetic and potential energy functions

"""

# import modules
import numpy as np
from qwave.units import s_cm, B_m, au_kg, j_ev, ev_h

#import internal modules
from qwave.utilities import pes

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

def calculate_potential(grid: np.ndarray, potential_func: str, fit_type: str) -> np.ndarray:
    """
    Calculate the Potential Energy Matirix.
    Parameters:
        grid_points: int
        grid: np.ndarray
        potential_func: str
        fit_type: str
    Returns:
        V: np.ndarray (diagonal matrix)
    """

    V = [ pes(potential_func, grid_point, fit_type) for grid_point in grid[:-1] ]

    return np.diag(V)


def calculate_HO_potential(frequency: float, grid: np.ndarray, mass: float) -> np.ndarray:
    """
    Calculate the potential energy matrix for a harmonic oscillator.
    Parameters:
        frequency: float
        grid_points: int
        grid: np.ndarray
        mass: float
    Returns:
        V: np.ndarray (diagonal matrix)
    """         

    V = [(2*np.pi**2)*(frequency**2)*(s_cm**2)*(grid_point**2)*\
        ((B_m)**2)*mass*(au_kg)*j_ev*ev_h for grid_point in grid[:-1] ]
    return np.diag(V)
