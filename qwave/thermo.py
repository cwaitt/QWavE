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
from scipy.interpolate import CubicSpline
import numpy as np

# load internal modules
from qwave.utilities import derivative
from qwave.units import kb_unit_dict


def free_A(q, temperature, kb):
    return -kb * temperature * np.log(q)


def free_A_S(partition_func, temperatures, unit):

    try:
        kb = kb_unit_dict[unit]
    except KeyError:
        raise ValueError('Unit must be Hartree, eV, J, or kJ/mol')


    A = free_A(partition_func, temperatures, kb)

    cubic_spline = CubicSpline(temperatures, A)

    S = np.array([-derivative(cubic_spline, temp) for temp in temperatures])

    return A, S

def free_A_p(partition_func: np.ndarray, temperatures, volumes, unit='J'):

    if unit == 'J':
        kb = kb_unit_dict[unit]
    else:
        raise ValueError('Module designed only if free energy is in Joules')

    A = free_A(partition_func, temperatures, kb)

    cubic_spline = CubicSpline(temperatures, A)

    p = np.array([derivative(cubic_spline,vol) for vol in volumes])

    return A, p

