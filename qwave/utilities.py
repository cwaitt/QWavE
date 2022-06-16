"""
utilities.py
Some simple functions used to simplify computation of the SE and stat mech formula

"""

# import modules
import numpy as np
from scipy import constants 
import csv
from scipy.interpolate import CubicSpline

# define functions

def ngrid(box_length: float, grid_points: float):
    grid = np.linspace(-1*box_length/2, box_length/2, grid_points+1)   # Defines grid to perfom numerical evaluation of SE
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

def sort_energy(eigval,len_eigval):
    indices = np.argsort(eigval)
    indices = indices[:len_eigval] # limits the number of eigenvalues
    energies = eigval[indices]
    
    return energies

def sort_wave(energy, eigval, wave):
    psi = []
    order = np.argsort(eigval)
    for i, energy_i in enumerate(energy):
        psi.append(wave[:, order[i]] + energy_i)  #Corresponding component of eigenvector (value of wavefunction)

    return psi

def derivative(f, a, method='central', h=0.1):
    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError('Method must be central, forward or backward')
