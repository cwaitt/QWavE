"""
This file will store unit dictionaries for different constants,
which will be used in calculations and unit conversions.
"""

from scipy.constants import physical_constants, N_A


kb_unit_dict = {'Hartree': physical_constants['Boltzmann constant in eV/K'][0]/physical_constants['Hartree energy in eV'][0],\
                'eV': physical_constants['Boltzmann constant in eV/K'][0],\
                'J': physical_constants['Boltzmann constant'][0],\
                'kJ/mol': physical_constants['Boltzmann constant'][0]/1000/N_A
                }

h_unit_dict = {'Hartree': -physical_constants['Planck constant'][0]/physical_constants['hartree-joule relationship'][0],\
                'eV': -physical_constants['Planck constant in eV s'][0],
                'J': -physical_constants['Planck constant'][0],
                'kJ/mol': -physical_constants['Planck constant'][0]/1000/N_A
                }


eV_to_J = physical_constants['electron volt-joule relationship'][0]
bohr_to_m = physical_constants['Bohr radius'][0]
kb_default = physical_constants['kelvin-hartree relationship'][0]

c = physical_constants['speed of light in vacuum'][0] * 100

s_cm = physical_constants['speed of light in vacuum'][0] * 100
B_m = physical_constants['Bohr radius'][0]
j_ev = physical_constants['joule-electron volt relationship'][0]
ev_h = physical_constants['electron volt-hartree relationship'][0]
am_kg = physical_constants['atomic mass constant'][0]
au_kg = physical_constants['atomic unit of mass'][0]

