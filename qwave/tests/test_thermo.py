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
from .test_utilities import *

def test_free_A_S(q,temp,unit):

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


    A = -1*kb*temp*np.log(q)

    cs = CubicSpline(temp,A)

    S = []
    for i in temp:
        S.append(derivative(cs,i)*-1)
    
    S = np.array(S)

    return A,S

def test_free_A_p(q,temp,volume,unit='J'):

    if unit == 'J':
        kb = constants.physical_constants['Boltzmann constant'][0]

    else:
        raise ValueError('Module designed only if free energy is in Joules')


    A = -1*kb*temp*np.log(q)

    cs = CubicSpline(temp,A)

    p = []
    for i in volume:
        p.append(derivative(cs,i)*-1)
    

    return A,p

   
