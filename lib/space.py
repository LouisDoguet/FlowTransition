import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import copy

class Space:
    def __init__(self,xlim,ylim,Nx,Ny=None):
        '''
        Create a 2D spatial grid for potential flow calculations.

        @param xlim: tuple (xmin, xmax) defining the x-axis limits
        @param ylim: tuple (ymin, ymax) defining the y-axis limits
        @param Nx: number of grid points in the x-direction
        @param Ny: number of grid points in the y-direction (optional; if None, Ny = Nx)
        '''
        if not Ny:
            Ny = Nx
        self.xmin = xlim[0]
        self.xmax = xlim[1]
        self.ymin = ylim[0]
        self.ymax = ylim[1]
        self.Nx = Nx
        self.Ny = Ny
        self._x = np.linspace(self.xmin,self.xmax,Nx)
        self._y = np.linspace(self.ymin,self.ymax,Ny)
        self.X, self.Y = np.meshgrid(self._x,self._y)
        self.Z = self.X+1j*self.Y

    def duplicate(self, Z):
        cp = copy.copy(self)
        cp.Z = Z
        return cp

    def plot(self,u,v,nfig=1,title='',density=3):
        '''
        Plot the given velocity field as streamlines.

        @param u: 2D array of x-velocity components
        @param v: 2D array of y-velocity components
        @param nfig: figure number
        @param title: title of the plot
        '''

        speed = np.sqrt(u**2 + v**2)

        fig = plt.figure(nfig)  # Reuse figure if it exists
        ax = fig.gca()
        ax.streamplot(self.X,self.Y,u,v,density=density,color=speed,linewidth=1,arrowsize=0.5,cmap='magma',broken_streamlines=True)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return ax
