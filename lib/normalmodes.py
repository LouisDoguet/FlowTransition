import matplotlib.pyplot as plt
import lib.space as sp
from lib.complexplot import ComPlot
import numpy as np

class LinNSSol:
    def __init__(self, function, alpha, omega, space, tarray):
        self._function = function
        self.alpha = alpha
        self.omega = omega
        self.space = space
        self.t = tarray
        self.sol = []

        X, Y = self.space.X, self.space.Y
        
        for tmstp in tarray:
            Z = self._function(Y) * np.exp(1j* (alpha*X - omega*tmstp) )
            self.sol.append(space.duplicate(Z))

    def animate(self, title='Normal Mode Evolution', dynamic_scaling=True):
        """
        Animate the time evolution of the normal mode solution.
        """
        cp = ComPlot()
        cp.fromSpace(self.space)
        cp.animate_fields(self.sol, title=title, dynamic_scaling=dynamic_scaling)

    def animate_3d(self, title='Normal Mode Evolution 3D', dynamic_scaling=True):
        """
        Animate the time evolution of the normal mode solution in 3D.
        """
        cp = ComPlot()
        cp.fromSpace(self.space)
        cp.animate_fields_3d(self.sol, title=title, dynamic_scaling=dynamic_scaling)

