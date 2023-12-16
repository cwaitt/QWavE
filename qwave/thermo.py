"""
thermo.py

Evaluates thermodynamic properties from probabilities and partition function
"""
# load modules
#from scipy.interpolate import CubicSpline
import numpy as np

# load internal modules
#from .utilities import derivative
from .units import kb_eh

def internal_U(ej: np.ndarray ,prob: np.ndarray):
    """
    Evaluate internal energy (average energy) 

    Parameters:
        ej: np.ndarray collection of eigenstants evaluated by the SE (hartree)
        prob: np.ndarray probabilies of associated eigenstates at temperature T

    Return:
        U: np.ndarray internal energy (hartree)
    """

    U = np.zeros(len(prob))

    U_temp = ej*prob # temporary variable to store data

    for i,u in enumerate(U_temp):
        U[i] = np.sum(u)

    return U

def entropy_S(prob):
    """
    Evaluate entropy (hartree/K)

    Parameters:
        prob: np.ndarray probabilities of associated eigenstates at temperature T

    Return:
        S: np.ndarray entropy (hartree/k)
    """

    S = np.zeros(len(prob))

    S_temp = -prob*np.log(prob)

    for i,s in enumerate(S_temp):
        S[i] = np.sum(s)*kb_eh

    return S

def helmholtz_F(part,temp):
    """
    Evaluates the Helmholtz Free energy

    Parameters:
        part: np.ndarray partition function at temperature T

    Return:
        F: np.ndarray Helmholtz Free Energy (hartree)
    """

    F = -kb_eh*temp*np.log(part)

    return F
    

#def free_A(q, temperature, kb):
#    return -kb * temperature * np.log(q)


#def free_A_S(partition_func, temperatures, unit):

#    try:
#        kb = kb_unit_dict[unit]
#    except KeyError:
#        raise ValueError('Unit must be Hartree, eV, J, or kJ/mol')


#    A = free_A(partition_func, temperatures, kb)

#    cubic_spline = CubicSpline(temperatures, A)

#    S = np.array([-derivative(cubic_spline, temp) for temp in temperatures])

#    return A, S

#def free_A_p(partition_func: np.ndarray, temperatures, volumes, unit='J'):

#    if unit == 'J':
#        kb = kb_unit_dict[unit]
#    else:
#        raise ValueError('Module designed only if free energy is in Joules')

#    A = free_A(partition_func, temperatures, kb)

#    cubic_spline = CubicSpline(temperatures, A)

#    p = np.array([derivative(cubic_spline,vol) for vol in volumes])

#    return A, p

