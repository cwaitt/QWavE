"""
partition_func.py
A statistcal mechanics solver to evaluate the parition function given a collection of eigen states

Handles the primary functions
ei:
    list of eigen values from schroginger equation
temp:
    array of temperatures to evaluate partition function
"""
# load modules
import numpy as np
from .units import kb_eh

def canonical_q(ej: np.ndarray, temp_arr: np.ndarray):
    """
    Evaluates the canonical partition function: q(N,V,T)
    
    Parameters:
        ej: np.ndarray collection of eigenstates evaluated by the SE
        temp_arr: np.ndarray temperature range to evaluate parition function
    Returns:
        Q: np.ndarray parition function
        Prob: np.ndarray 
    """
    
    temp = np.array(temp_arr)
    
    Q = np.zeros(len(temp)) #store partition function
    Prob = np.zeros((len(temp),len(ej))) #store probabilities
    
    for i,t in enumerate(temp):
        q = np.sum(np.exp(-ej/(kb_eh*t)))
        if q == 0: # Sets probabilities of first state to 1 and all others zero for low temperatures if the partition function is equal to 0
            prob = np.zeros(len(ej))
            prob[0] = 1
            if i == 0:
                print("""Temperatures are too small which may lead to numerical errors. Consider increasing the temperature. Use the values obtained at your own risk.""")
        else:
            prob = np.exp(-ej/(kb_eh*t))/q
        
        Q[i] = q
        Prob[i] = prob
        
    return Q,Prob # return partition function value at each temperature
                  # and probability of occupying each state at each temperature
