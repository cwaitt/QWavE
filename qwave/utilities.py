"""
utilities.py
Some simple functions used to simplify computation of the SE and stat mech formula

"""

# import modules
import numpy as np
import csv
from scipy.interpolate import CubicSpline

# define functions

def ngrid(box_length: float, grid_points: int):
    """
    Calculate the number of grid points along one axis.
    Parameters:
        box_length: float
        grid_points: int
    Returns:
        grid: np.ndarray
        dgrid: float
    """

    grid = np.linspace(-box_length/2, box_length/2, grid_points + 1)   # Defines grid to perfom numerical evaluation of SE
    dgrid = grid[1] - grid[0]                                  # Distance between grid points

    return grid, dgrid

def pes(potential_func, grid, fit_type='not-a-knot'):

    pot_exp = None
    if potential_func == 'PIAB':
        pot_exp = 0

    elif potential_func == 'PARA':
        pot_exp = 0.5*grid**2

    else:
        xdata = []
        ydata = []

        with open(potential_func) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')

            for row in csv_reader:
                xdata = np.append(xdata,float(row[0]))
                ydata = np.append(ydata, float(row[1]))

        cubic_spline = CubicSpline(xdata,ydata,bc_type=fit_type)

        pot_exp = cubic_spline(grid)

    return pot_exp

def sort_energy(eigval: np.ndarray, len_eigval: int) -> np.ndarray:
    """
    Sort the eigenvalues in ascending order.
    Parameters:
        eigval: np.ndarray
        len_eigval: int
    Returns:
        energies: np.ndarray
    """
    energies = np.sort(eigval)[:len_eigval]
    return energies

def sort_wave(energy, eigval, wave):
    """
    Sort the eigenfunctions in ascending order (of energy).
    Parameters:
        energy: np.ndarray
        eigval: np.ndarray
        wave: np.ndarray
    Returns:
        psi: list of np.ndarray wavefunctions
    """
    psi = []
    order = np.argsort(eigval)
    for i, energy_i in enumerate(energy):
        psi.append(wave[:, order[i]] + energy_i)  #Corresponding component of eigenvector (value of wavefunction)

    return psi

def derivative(f, a: float, method='central', h=0.1) -> float:
    """
    Calculate the derivative of a function using a difference method.
    Parameters:
        f: function
        a: float
        method: str
        h: float
    Returns:
        result: float
    """

    if method == 'central':
        result = (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        result = (f(a + h) - f(a))/h
    elif method == 'backward':
        result = (f(a) - f(a - h))/h
    else:
        raise ValueError('Method must be central, forward or backward')
    return result
