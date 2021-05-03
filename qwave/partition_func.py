"""
partition_func.py
A statistcal mechanics solver to evaluate the parition function given a collection of eigen states

Handles the primary functions
ei: 
    list of eigen values from schroginger equation
temp: 
    array of temperatures to evaluate partition function
unit:
    determines the units of boltzmann constant (must be same as eigen values)
        kJ/mol
        J
        eV
plot (optional):
    function that can be set to True to plot out the solutions to the SE
"""
# load modules
from scipy import constants
import numpy as np

#load internal modules
from .plot import *

eV_to_J = constants.physical_constants['electron volt-joule relationship'][0]

bohr_to_m = constants.physical_constants['Bohr radius'][0]

def q_PESq(ei,temp,unit):

    if unit == 'Hartree':
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]/constants.physical_constants['Hartree energy in eV'][0]
    elif unit == 'eV':
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
    elif unit == 'J':
        kb = constants.physical_constants['Boltzmann constant'][0]
    elif unit == 'kJ/mol':
        kb = constants.physical_constants['Boltzmann constant'][0]/1000*constants.N_A
    else:
        raise ValueError('Unit must be Hartree, eV, J, or kJ/mol')

    q_tot = []

    for i in temp:
        q_temp = 0
        for j in ei:
            q_temp = np.exp(-j/(kb*i)) + q_temp

        q_tot.append(q_temp)

    q_tot=np.array(q_tot)

    return q_tot

def q_HO(freq,temp,unit):

    c = constants.physical_constants['speed of light in vacuum'][0]*100

    if unit == 'Hartree':
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]/constants.physical_constants['Hartree energy in eV'][0]
        h = -1*constants.physical_constants['Planck constant'][0]/constants.physical_constants['hartree-joule relationship'][0]
    elif unit == 'eV':
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
        h = -1*constants.physical_constants['Planck constant in eV s'][0]
    elif unit == 'J':
        kb = constants.physical_constants['Boltzmann constant'][0]  
        h = -1*constants.physical_constants['Planck constant'][0]
    elif unit == 'kJ/mol':
        kb = constants.physical_constants['Boltzmann constant'][0]/1000*constants.N_A
        h = -1*constants.physical_constants['Planck constant'][0]/1000*constants.N_A
    else:
        raise ValueError('Unit must be Hartree, eV, J, or kJ/mol')


    q_tot = []

    for i in temp:
        q_temp = 0

        beta = kb*i
        en_freq = h*freq*c

        q_temp = np.exp(en_freq/(2*beta)) / (1-np.exp(en_freq/beta))

        q_tot.append(q_temp)

    q_tot = np.array(q_tot)

    return q_tot

def q_HT(Vb,ax,mass,temp,unit):

    if unit == 'eV':
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
        h = constants.physical_constants['Planck constant in eV s'][0]
    else:
        raise ValueError('Unit must be eV')

    print('Warning: make sure mass is in kg and ax is in bohr')

    vx = ((Vb*eV_to_J)/(2*mass*(ax*bohr_to_m)**2))**0.5 # freqeuncy

    rx = Vb/(h*vx)

    Tx = (kb*temp)/(h*vx)

    qclass = []

    qHO = []

    qzpe = []

    for j in Tx:

        qclass.append(np.sqrt((np.pi*rx/j))*np.exp(-1*rx/(2*j))*(np.i0(rx/(2*j))))

        qHO.append(np.exp(-1/(2*j))/(1-np.exp(-1/j)))

        qzpe.append(np.exp(1/((2+(16*rx))*j)))

    q_tot = np.array(qclass)*np.array(qHO)*np.array(qzpe)

    return q_tot

def q_rot(sigma, Ia, Ib, Ic, temp, unit):
    
    if unit == 'Hartree':
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]/constants.physical_constants['Hartree energy in eV'][0]
        h = -1*constants.physical_constants['Planck constant'][0]/constants.physical_constants['hartree-joule relationship'][0]
    elif unit == 'eV':
        kb = constants.physical_constants['Boltzmann constant in eV/K'][0]
        h = -1*constants.physical_constants['Planck constant in eV s'][0]
    elif unit == 'J':
        kb = constants.physical_constants['Boltzmann constant'][0]  
        h = -1*constants.physical_constants['Planck constant'][0]
    elif unit == 'kJ/mol':
        kb = constants.physical_constants['Boltzmann constant'][0]/1000*constants.N_A
        h = -1*constants.physical_constants['Planck constant'][0]/1000*constants.N_A
    else:
        raise ValueError('Unit must be Hartree, eV, J, or kJ/mol')
    
    q_rot = []
    
    for i in temp:
        print(i)
        q_temp = 0
        
        coeff = ((8 * (np.pi**2) * kb * i) / (h**2))
    
        term1 = np.sqrt(np.pi) / sigma
        term2 = np.sqrt(coeff * Ia)
        term3 = np.sqrt(coeff * Ib)
        term4 = np.sqrt(coeff * Ic)
    
        q_temp = term1 * term2 * term3 * term4
        
        q_rot = np.append(q_rot, q_temp)
    
    return q_rot
