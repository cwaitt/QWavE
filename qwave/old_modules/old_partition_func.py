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
#from qwave.units import kb_unit_dict, h_unit_dict, eV_to_J, c, bohr_to_m
from .units import kb_unit_dict, h_unit_dict, eV_to_J, c, bohr_to_m

def q_PESq(ei, temperatures, kb):

    q_tot = []

    for temperature in temperatures:
        q_temp = 0
        beta = 1/(kb * temperature)
        for energy in ei:
            q_temp += np.exp(-beta * energy)

        q_tot.append(q_temp)

    q_tot=np.array(q_tot)

    return q_tot

def q_HO(frequency, temperatures, kb, h):

    log_q_tot = []

    for temperature in temperatures:
        # q_temp = 0

        beta = kb * temperature
        en_freq = -h * frequency * c

        log_q_temp = en_freq /(2*beta) - np.log(1-np.exp(en_freq/beta))
        log_q_tot.append(log_q_temp)

    log_q_tot = np.array(log_q_tot)
    log_q_mean = np.mean(log_q_tot)
    log_q_tot -= log_q_mean # This will help with the overflow errors
    q_tot = np.exp(log_q_tot) * np.exp(log_q_mean)

    return q_tot

def q_HT(Vb,ax,mass,temp,unit):

    if unit == 'eV':
        kb = kb_unit_dict['eV']
        h = h_unit_dict['eV']
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

def q_rot(sigma, Ia, Ib, Ic, temperatures, unit):

    try:
        kb = kb_unit_dict[unit]
        h = h_unit_dict[unit]

    except KeyError:
        raise ValueError('Unit must be Hartree, eV, J, or kJ/mol')
    
    q_rot_list = []
    
    for temperature in temperatures:
        
        coeff = ((8 * (np.pi**2) * kb * temperature) / (h**2))
    
        term1 = np.sqrt(np.pi) / sigma
        term2 = np.sqrt(coeff * Ia)
        term3 = np.sqrt(coeff * Ib)
        term4 = np.sqrt(coeff * Ic)
    
        q_temp = term1 * term2 * term3 * term4
        
        q_rot_list.append(q_temp)
    
    return q_rot_list
