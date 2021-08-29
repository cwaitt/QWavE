"""
statistics.py
A python module that evaluates the Boltzmann statistics from the partition function

eig:
    array of eigenvalues from the schrodinger equation

part:
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
from .utilities import *

def bolt_prob(eig,num,part,temp,kb):

    state = np.exp(-eig[num]/(kb*temp))
    prob = state/part

    return prob

def avg_energy(part,temp,kb):

    beta = np.flipud(1/(kb*temp))
    lnq = np.flipud(np.log(part))
    cs = CubicSpline(beta,lnq)

    E = np.flipud(-1*derivative(cs,beta))
    
    cs = CubicSpline(temp,E)

    cv = derivative(cs,temp)

    variance = cv * kb*(temp**2)

    return E, variance, cv

