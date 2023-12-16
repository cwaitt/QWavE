"""
statistics.py
A python module that evaluates the Boltzmann statistics from the partition function

energy:
    selected eigenvalue from the schrodinger equation

partition_function:
    array corresponding to the value of the partition function at temperature T

temp:
    array of temperatures

kb:
    Boltzmann constant
"""

# load modules
import numpy as np
from scipy.interpolate import CubicSpline

# load internal modules
from .utilities import derivative

def boltzmann_probability(energy: float, partition_function: float, temperature: float, kb: float) -> float:
    """
    Calculate the probability of a particular state based on its energy,
    given the Boltzmann distribution.
    Parameters
    ----------
    energy : float
        Energy of the state.
    partition_function : float
        Partition function of the system.
    temperature : float
        Temperature of the system.
    kb : float
        Boltzmann constant (in same units as energy).
    """

    beta = 1/(kb * temperature)

    state = np.exp(-beta * energy)
    prob = state/partition_function

    return prob

def average_energy(partition_function: np.ndarray, temperature: float, kb: float):
    """
    Calculate the average energy of a system by using the partition function.
    Parameters
    ----------
    partition_function : float
        Partition function of the system.
    temperature : float
        Temperature of the system.
    kb : float
        Boltzmann constant (in same units as energy).
    """

    beta = np.flipud(1/(kb*temperature))
    lnq = np.flipud(np.log(partition_function))
    cs = CubicSpline(beta,lnq)

    E = np.flipud(-derivative(cs,beta))
    
    cs = CubicSpline(temperature,E)

    cv = derivative(cs,temperature)

    variance = cv * kb*(temperature**2)

    return E, variance, cv
