"""
schrodinger.py
A SE solver for models such as the particle in the box and complicated models for an arbitrary potential

Handles the primary functions
box_length:
    length of the box to evaluate the SE (must be in a.u.)
mass:
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
problem:
    specify which problem to be solved, currently supports:
        particle in a box: 'box'
        Harmonic oscillator: 'HO'
"""
from numpy.linalg import eig
# load internal modules
from qwave.hamiltonian import calculate_kinetic, calculate_HO_potential, calculate_potential
from qwave.utilities import ngrid, sort_energy, sort_wave


def schrodinger_solution(box_length: int, mass: float, pot_func='PIAB', fit_type='not-a-knot',
        grid_points=101, len_eigval=10, problem=None, frequency=None):

    """
    Calculate the energies and corresponding wavefunctions for the Schrodinger equation.
    Parameters:
        box_length: float
        mass: float
        pot_func: str
        fit_type: str
        grid_points: int
        len_eigval: int
        problem: str ('box' or 'HO')
        frequency: float (required for 'HO')
    Returns:
        energies: np.ndarray
        wavefunctions: np.ndarray
    """

    grid, dgrid = ngrid(box_length, grid_points)

    C = -1/(2 * mass * dgrid**2)                        # evaluate the constant of the kinetic energy operator

    if 'box' in problem.lower():
        V = calculate_potential(grid, pot_func, fit_type)  # evaluate the potential energy operator

    elif 'HO' in problem:
        V = calculate_HO_potential(frequency, grid, mass)

    else:
        raise NotImplementedError(f'{problem} is not a valid problem')
    T = calculate_kinetic(grid_points)                           # evaluate the kinetic energy operator

    H = (C*T) + V

    eigval, eigvec = eig(H)                   # Diagonalize Hamiltonian for eigenvectors

    energy = sort_energy(eigval, len_eigval)             # Sort the values from lowest to highest
    wavefunc = sort_wave(energy, eigval, eigvec)

    return energy, wavefunc
