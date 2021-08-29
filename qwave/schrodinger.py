"""
schrodinger.py
A SE solver for models such as the particle in the box and complicated models for an arbitrary potential

Handles the primary functions
box_length: 
    length of the box to evaluate the SE (must be in a.u.)
mass 
    mass of the particle in the box (must be in a.u.)
pot_func:
    supply your own potential energy functional or use a preset functional (default = Particle in a Box (PIAB))
    preset potentials include: 
        zero potential (PIAB)
        parabola (PARA)
len_eigval:
    number of eigenvalues you would like to compute (default = 10)
grid_points (optional):
    generates N+1 (default 101) gridpoints to numerically evaluate the SE
unit (optional):
    if set will convert energies to units from atomic (default = None)
    availible options:
        kJ/mol
        J
        eV
plot (optional):
    function that can be set to True to plot out the solutions to the SE with respesct to the potential
"""

# load internal modules
from .utilities import *
from .hamiltonian import *

def schrodinger_box(box_length,mass,pot_func='PIAB',fit_type='not-a-knot',
        grid_points=101,len_eigval=10):

    grid,dgrid = ngrid(box_length,grid_points)

    C = -1/(2*mass*dgrid**2)                        # evaluate the constant of the kinetic energy operator

    V = eval_pot(grid_points,grid,box_length,pot_func,fit_type)  # evaluate the potential energy operator

    T = eval_kin(grid_points)                           # evaluate the kinetic energy operator

    H = (C*T) + V

    eigval, eigvec = np.linalg.eig(H)                   # Diagonalize Hamiltonian for eigenvectors

    energy = sort_energy(eigval,len_eigval)             # Sort the values from lowest to highest 
    wavefunc = sort_wave(energy,eigval,eigvec)

    return energy, wavefunc
  
def schrodinger_HO(box_length,mass,frequency,
        grid_points=101,len_eigval=10):

    grid,dgrid = ngrid(box_length,grid_points)

    C = -1/(2*mass*dgrid**2)

    V = eval_pot_HO(frequency,grid_points,grid,mass)

    T = eval_kin(grid_points)

    H = (C*T) + V

    eigval, eigvec = np.linalg.eig(H)

    energy = sort_energy(eigval,len_eigval)
    wavefunc = sort_wave(energy,eigval,eigvec)

    return np.array(energy), np.array(wavefunc)


