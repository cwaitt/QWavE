"""
hamiltonian.py

Functions to evaluate the kinetic and potential energy functions

"""

# import modules
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import spdiags, kronsum, diags
from scipy.interpolate import CubicSpline, bisplrep, bisplev
import pandas as pd

def calc_kinetic_1D(grid_points: int):
    
    """
    Calculate the 1D Kinetic Energy Matirix.
    Utilizes fourth-ordered central difference approximation.
    Parameters:
        grid_points: int
    Returns:
        T np.ndarray
    """
    T = np.zeros((grid_points, grid_points)) # initialize matrix of zeros
    
    for i in range(grid_points): # add 4th-ordered central difference method to diagnals of matrix
        T[i,i] = -30/12
        for j in range(grid_points):
            if abs(i - j) == 1:
                T[i,j] = 16/12
            elif abs(i-j) == 2:
                T[i,j] = -1/12
            else:
                pass # keep zeros everywhere else
        
    return T

def calc_kinetic_2D(grid_points: int):
    
    """
    Calculate the 2D Kinetic Energy Matirix.
    Utilizes fourth-ordered central difference approximation.
    Parameters:
        grid_points: int
    Returns:
        T: scipy.sparse._dia.dia_matrix
    """
    diag_1 = np.repeat(-30/12,grid_points) # create an array of diagnal elements
    diag_2 = np.repeat(16/12,grid_points)
    diag_3 = np.repeat(-1/12,grid_points)
    diags = np.array([diag_3,diag_2,diag_1,diag_2,diag_3]) # organize diagnal elements (read from center)
    D = spdiags(diags,np.array([-2,-1,0,1,2]), grid_points, grid_points) # specify order of diagonal and off diagnol elements in sparse matrix
    
    T = kronsum(D,D) # kroneker sum matrices
        
    return T

def calc_potential_1D(grid_points: int, grid: np.ndarray, pot_func: str):
    
    """
    Calculate the Potential Energy Matirix.
    Parameters:
        grid: np.ndarray
        pot_func: str or path to formatted csv file
                piab: particle in a box
                para: particle in a parabolic well
                tunn: particle in a piecewise potential
    Returns:
        V: np.ndarray (diagonal matrix)
        pot_exp: potential energy expression (to plot with eigenvalues and wavefunctions)
    """
    
    pot_expression = None
    
    if pot_func.lower() == 'piab': # define potentials for comparison with analytical solutions
        pot_exp = 0*grid
        
    elif pot_func.lower() == 'para':
        pot_exp = 0.5*grid**2
        
    elif pot_func.lower() == 'tunn':
        pot_exp_left = grid[0:int(np.round(grid_points/2))]*0
        pot_exp_right = grid[int(np.round(grid_points/2)):]*0 + 3
        pot_exp = np.concatenate((pot_exp_left,pot_exp_right))
        
    else:
        csv = pd.read_csv(pot_func) # cubic spline arbitrary potential (that satifies the condition Psi(lx/2)=Psi(-lx/2)=0)
        xdata = np.array(csv['x'])
        ydata = np.array(csv['y'])
                
        cubic_spline = CubicSpline(xdata,ydata,bc_type='not-a-knot')      
        pot_exp = cubic_spline(grid)
        
    V = np.diag(pot_exp) # diagonlize potential
    
    return V, pot_exp

def calc_potential_2D(grid_points: int, Xgrid: np.ndarray, Ygrid: np.ndarray, pot_func: str):
    
    """
    Calculate the Potential Energy Matirix.
    Parameters:
        grid: np.ndarray
        pot_func: str or path to formatted csv file
                piab: particle in a box
                para: particle in a parabolic well
                tunn: particle in a piecewise potential
    Returns:
        V: np.ndarray (diagonal matrix)
    """
    
    pot_expression = None
    
    if pot_func.lower() == 'piab':
        pot_exp = 0*Xgrid + 0*Ygrid
        
    elif pot_func.lower() == 'para':
        pot_exp = 0.5*Xgrid**2 + 0.5*Ygrid**2
        
    else:
        csv = pd.read_csv(pot_func) # cubic spline arbitrary potential (that satifies the condition Psi(lx/2)=Psi(-lx/2)=0)
        xdata = np.array(csv['x'])
        ydata = np.array(csv['y'])
        zdata = np.array(csv['z'])
        
        data_n = int(np.sqrt(len(xdata))) # square root of the number of data points in each column
        
        # grid input data
        xgrid = xdata.reshape(data_n,data_n)
        ygrid = ydata.reshape(data_n,data_n)
        zgrid = zdata.reshape(data_n,data_n)
        

        spline = bisplrep(xgrid, ygrid, zgrid) # spline data
        pot_exp = bisplev(Xgrid[0,:], Ygrid[:,0], spline)
   
    V = diags(pot_exp.reshape(grid_points**2),(0)) # diagonlize potential
    
    return V, pot_exp


def calculate_HO_potential(frequency: float, grid: np.ndarray, mass: float) -> np.ndarray:
    """
    Calculate the potential energy matrix for a harmonic oscillator.
    Parameters:
        frequency: float
        grid_points: int
        grid: np.ndarray
        mass: float
    Returns:
        V: np.ndarray (diagonal matrix)
    """         

    V = [(2*np.pi**2)*(frequency**2)*(s_cm**2)*(grid_point**2)*\
        ((B_m)**2)*mass*(au_kg)*j_ev*ev_h for grid_point in grid[:-1] ]
    return np.diag(V)
