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

def test_ngrid(box_length,grid_points):
    grid = np.linspace(-1*box_length/2,box_length/2,grid_points+1)   # Defines grid to perfom numerical evaluation of SE
    dgrid = grid[1] - grid[0]                                  # Distance between grid points

    return grid, dgrid

def test_pes(pot_func,grid,box_length,fit_type='not-a-knot'):
    global pot_exp
    if pot_func == 'PIAB':
        pot_exp = 0

    elif pot_func == 'PARA':
        pot_exp = 0.5*grid**2

    else:
        xdata = []
        ydata = []

        with open(pot_func) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            line_count = 0
            for row in csv_reader:
                xdata = np.append(xdata,float(row[0]))
                ydata = np.append(ydata, float(row[1]))

        cs = CubicSpline(xdata,ydata,bc_type=fit_type)

        pot_exp = cs(grid)

    return pot_exp

def test_sort_energy(eigval,len_eigval):
    indices = np.argsort(eigval)
    indices = indices[:len_eigval] # limits the number of eigenvalues
    energies = eigval[indices]
    
    return energies

def test_sort_wave(energy,eigval,wave):
    psi = []
    order = np.argsort(eigval)
    for i in range(len(energy)):
        psi.append(wave[:, order[i]] + energy[i])  #Corresponding component of eigenvector (value of wavefunction)

    return psi

def test_energy_conv(energy,unit):
    if unit == 'J':
        new_e = energy * constants.physical_constants['Hartree energy'][0]
    elif unit =='kJ/mol':
        new_e = energy * constants.physical_constants['Hartree energy'][0]/1000/constants.N_A
    elif unit == 'eV':
        new_e = energy * constants.physical_constants['Hartree energy in eV'][0]
    else:
        raise ValueError('Unit must be J, kJ/mol, or eV')

    return new_e

def test_derivative(f,a,method='central',h=0.1):
    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError('Method must be central, forward or backward')
