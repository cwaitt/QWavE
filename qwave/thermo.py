"""
partition_func.py
A statistcal mechanics solver to evaluate the parition function given a collection of eigen states

Handles the primary functions
q:
    list of eigen values from schroginger equation
temp:
    array of temperatures to evaluate partition function
volume:
    array of volumes to evaluate partition function
unit:
    determines the units of boltzmann constant (must be same as eigen values)
        kJ/mol
        J
        eV
"""
# load modules
from scipy import constants
from scipy.interpolate import CubicSpline

# load internal modules
from .utilities import *

kb_default = constants.physical_constants['kelvin-hartree relationship'][0]

def free_A(q, temperature, kb):
    return -kb * temperature * np.log(q)


def free_A_S(q, temperature, unit):

    if unit == 'Hartree':
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]/constants.physical_constants['Hartree energy in eV'][0]

    elif unit == 'eV':
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]

    elif unit == 'J':
        kb = constants.physical_constants['Boltzmann constant'][0]

    elif unit == 'kJ/mol':
        kb = constants.physical_constants['Boltzmann constant'][0]/1000/constants.N_A

    else:
        raise ValueError('Unit must be Hartree, eV, J, or kJ/mol')


    A = free_A(q, temperature, kb)

    cubic_spline = CubicSpline(temperature, A)

    S = np.array([-derivative(cubic_spline, temp) for temp in temperature])

    return A, S

def free_A_p(q, temperature, volume, unit='J'):

    if unit == 'J':
        kb = constants.physical_constants['Boltzmann constant'][0]
    else:
        raise ValueError('Module designed only if free energy is in Joules')

    A = free_A(q, temperature, kb)

    cubic_spline = CubicSpline(temperature, A)

    p = np.array([derivative(cubic_spline,vol) for vol in volume])

    return A, p

   
