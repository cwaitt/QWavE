"""
plot.py
Some functions to visualize results of the SE and plot energies and entropies from stat mech

"""

# import modules
import numpy as np
import matplotlib.pyplot as plt

#def plot_box(grid,energy,boxlength)
    

def test_plot_se(grid,energy,wavefunc,V,box_length):
    plt.plot(np.linspace(-box_length/2,box_length/2,len(grid)-1),np.diag(V),'b-')
    for i in range(len(energy)):
        energy_plot = np.repeat(energy[i],len(grid))
        plt.plot(grid,energy_plot,'-',color='black')

    #plt.ylim(-0.01,energy[-1])
    plt.show()


