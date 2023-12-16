"""
schrodinger.py
A SE solver for models such as the particle in the box and complicated models for an arbitrary potential where analytical solutions are hard to obtain.

Contains the primary functions to evaluate the analytical solutions to the 1,2,and 3D partilce in a box (PIAB), and numerical solver for 1 and 2D SE

"""

# import standard python modules
import numpy as np
from scipy.sparse.linalg import eigsh
import time

# load internal modules
from .hamiltonian import *
#from qwave.utilities import ngrid, sort_energy, sort_wave

"""
Analytical functions for a PIAB

"""

def piab_1D(mass: float, length: float, n_x: int, lingrid = 101):
    
    """
    Calculate the analytical solution to the 1D SE for a particle in a box
    
    Parameters:
        mass: mass of particle (au)
        length: length of box (bohr)
        n_x: energy level (unitless)
        lingrid: density of grid to evaluate wavefunctions (wf)
        
    Returns:
        Ex : energy of particle (Hartree)
        Wave: wavefunction (arbitrary units)
        line: grid of points to plot wavefunction (bohr)
    """
    
    # check inputs
    if not isinstance(mass, float) or mass <=0:
        raise TypeError('mass must be a postive float')
    if not isinstance(length, float) or length <=0:
        raise TypeError('length must be a postive float')
    if not isinstance(n_x, int) or n_x < 1:
        raise TypeError('n_x must be a positive integer >= 1')
    
    # Evaluate analytical solution to 1D PIAB
    Ex = (n_x**2*np.pi**2)/((2*mass*length**2))
    
    # Evaluate wavefunction
    line = np.linspace(0,length,lingrid) # define gride to evaluate wf on
    Wave = np.sqrt(2/length)*np.sin(n_x*np.pi*line/length)
    
    return Ex, Wave, line

def piab_2D(mass: float, lx: float, ly: float, n_x: int, n_y: int, lingrid = 200):
    
    """
    Calculate the analytical solution to the 2D SE for a particle in a box
    
    Parameters:
        mass: mass of particle (au)
        lx: length of box in the x direction (bohr)
        ly: length of box in the y direction (bohr)
        n_x: energy state x (unitless)
        n_y: energy state y (unitless)
        lingrid: step size of grid to plot wavefunctions
        
    Returns:
        Exy : energy of particle (J)
        Wave: wavefunction (arbitrary units)
        linex: grid of points to plot wavefunction (m)
        liney: grid of points to plot wavefunction (m)
    """
    
    # check inputs
    if not isinstance(mass, float) or mass <=0:
        raise TypeError('mass must be a postive float')
    if not isinstance(lx, float) or lx <=0:
        raise TypeError('length must be a postive float')
    if not isinstance(ly, float) or ly <=0:
        raise TypeError('length must be a postive float')
    if not isinstance(n_x, int) or n_x < 1:
        raise TypeError('n_x must be a positive integer >= 1')
    if not isinstance(n_y, int) or n_y < 1:
        raise TypeError('n_y must be a positive integer >= 1')
    
    # Evaluate analytical solution to 2D PIAB
    Ex = ((n_x**2*np.pi**2)/(2*mass*lx**2)) 
    Ey = ((n_y**2*np.pi**2)/(2*mass*ly**2))
    Exy = Ex + Ey
    
    # Evaluate wavefunction
    linex = np.linspace(0,lx,lingrid)
    liney = np.linspace(0,ly,lingrid)
    
    Wave = []
    for l1 in linex:
        for l2 in liney:
            wave_temp = np.sqrt(2/lx)*np.sqrt(2/ly)*np.sin(n_x*np.pi*l1/lx)*np.sin(n_y*np.pi*l2/ly)
            Wave.append(wave_temp)
            
    Wave = np.array(Wave)
    Wave = Wave.reshape(lingrid,lingrid)
    
    return Exy, Wave, linex, liney

def piab_3D(mass: float, lx: float, ly: float, lz: float, n_x: int, n_y: int, n_z: int, lingrid = 200):
    
    """
    Calculate the analytical solution to the SE for a particle in a box
    
    Parameters:
        mass: mass of particle (kg)
        lx: length of box in the x direction (m)
        ly: length of box in the y direction (m)
        lz: length of box in the y direction (m)
        n_x: energy state x (unitless)
        n_y: energy state y (unitless)
        n_z: energy state z (unitless)
        lingrid: step size of grid to plot wavefunctions
        
    Returns:
        Exyz : energy of particle (J)
    """
    
    # check inputs
    if not isinstance(mass, float) or mass <=0:
        raise TypeError('mass must be a postive float')
    if not isinstance(lx, float) or lx <=0:
        raise TypeError('length must be a postive float')
    if not isinstance(ly, float) or ly <=0:
        raise TypeError('length must be a postive float')
    if not isinstance(lz, float) or lz <=0:
        raise TypeError('length must be a postive float')
    if not isinstance(n_x, int) or n_x < 1:
        raise TypeError('n_x must be a positive integer >= 1')
    if not isinstance(n_y, int) or n_y < 1:
        raise TypeError('n_y must be a positive integer >= 1')
    if not isinstance(n_z, int) or n_y < 1:
        raise TypeError('n_z must be a positive integer >= 1')
    
    # Evaluate analytical solution to 3D PIAB
    Ex = ((n_x**2*np.pi**2)/(2*mass*lx**2)) 
    Ey = ((n_y**2*np.pi**2)/(2*mass*ly**2))
    Ez = ((n_z**2*np.pi**2)/(2*mass*lz**2))
    Exyz = Ex + Ey + Ez
    
    return Exyz

"""
Numerical Solver for 1D and 2D SE

"""

def schrsol_1d(mass: float, lx: float, pot_func: str, grid_points=101, eig_len = 10):
    
    """
    Calculate the numerical solutions to the 1D-SE for a particle in an arbititrary potential
    
    Parameters:
        mass: mass of particle (au)
        lx: length of box in the x direction (bohr)
        pot_func: str or path to formatted csv file 
                    'piab' --> particle in a box
                    'para' --> particle in a parabola
                    'path/to/csv' --> file containing arbitrary potential 
        grid_points: density of grid (default 101)
        eig_len: maximum number of eigenvalues to report (default 10)
        
    Returns:
        energies: eigenstates of the SE (Hartree)
        psi: wavefunctions for corresponding eigen states (au)
        pot_e: potential to plot
    """
    
    # check inputs
    if not isinstance(mass, float) or mass <=0:
        raise TypeError('mass must be a postive float')
    if not isinstance(lx, float) or lx <=0:
        raise TypeError('length must be a postive float')
    if not isinstance(pot_func, str):
        raise TypeError('unidetifiable pot_func selected. Reverting to a PIAB')
        pot_func = 'piab'
    
    # Define grid to evaluate SE
    grid = np.linspace(-lx/2, lx/2, grid_points) # grid for finite (central) differentiation
    dgrid = grid[1] - grid[0] # spacing between grid points (dx)
    
    print('Computing Kinetic Energy Matrix')
    print(' ')
    T = calc_kinetic_1D(grid_points) # get kinetic energy matrix

    C = -1/(2 * mass * dgrid**2) # constant for kinetic energy
    
    print('Computing Potential Energy Matrix')
    print(' ')
    V, pot_e = calc_potential_1D(grid_points,grid,pot_func) # get potential energy matrix
    
    H = (C*T) + V # evaluate hamiltonian
    
    # Get eignvalues and eigenfunctions
    print('Evaluating Hamiltonian to obtain the {0} lowest eigenvalues and corresponding wavefunctions'.format(eig_len))
    print('Depending on your grid size this may take a few minutes')
    print(' ')
    eigval, eigvec = eigsh(H,k=eig_len,which='SM') # order by smallest values
    energies = eigval
    psi = eigvec.T

    print('Done')
    
    return energies, psi, pot_e

def schrsol_2d(mass: float, lx: float, ly: float, pot_func: str, grid_points=101, eig_len = 10):
    
    """
    Calculate the numerical solutions to the 2D-SE for a particle in an arbititrary potential
    
    Parameters:
        mass: mass of particle (au)
        lx: length of box in the x direction (bohr)
        ly: length of box in the x direction (bohr)
        pot_func: str or path to formatted csv file
                    'piab' --> particle in a box
                    'para' --> particle in a parabola
                    'path/to/csv' --> file containing arbitrary potential 
        grid_points: density of grid (default 101)
        eig_len: maximum number of eigenvalues to report (default 10)
        
    Returns:
        energies: eigenstates of the SE (Hartree)
        psi: wavefunctions for corresponding eigen states (au)
    """

    # measure time passed
    start = time.time()

    # Define grid to evaluate SE
    xgrid = np.linspace(-lx/2, lx/2, grid_points) # xgrid for finite (central) differentiation
    ygrid = np.linspace(-ly/2, ly/2, grid_points) # ygrid for finite (central) differentiation
    dx = xgrid[1] - xgrid[0] # spacing between grid points (dx)
    dy = ygrid[1] - ygrid[0] # spacing between grid points (dx)
    
    Xgrid,Ygrid = np.meshgrid(xgrid,ygrid) # discritize grid

    print('Computing Kinetic Energy Matrix')
    print(' ')
    T = calc_kinetic_2D(grid_points) # get kinetic energy matrix
    C = -1/(2*mass*dx*dy) # constant for kinetic energy

    print('Computing Potential Energy Matrix')
    print(' ')
    V, pot_e = calc_potential_2D(grid_points,Xgrid,Ygrid,pot_func) # get potential energy matrix
    
    H = (C*T) + V # evaluate hamiltonian
    
    #Get eignvalues and eigenfunctions
    print('Evaluating Hamiltonian to obtain the {0} lowest eigenvalues and corresponding wavefunctions'.format(eig_len))
    print('Depending on your grid size this may take a few minutes')
    print(' ')
    eigval, eigvec = eigsh(H,k=eig_len,which='SM') # order by smallest values
    energies = eigval
    psi = eigvec.T

    end = time.time()

    print('Done in {0:.2f} seconds'.format(end-start))
    
    return energies, psi, pot_e
