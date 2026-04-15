import numpy as np
from abc import ABC, abstractmethod


class BaseFlow(ABC):
    '''
    Abstract base class for parallel shear flow profiles U(y) on y ∈ [-1, 1].

    Subclasses must implement U(y) and Uyy(y).
    The wall-normal coordinate y is normalised so that solid walls sit at y = ±1.
    '''

    @property
    @abstractmethod
    def name(self):
        '''Human-readable name of the flow profile.'''
        pass

    @abstractmethod
    def U(self, y):
        '''
        Base flow velocity profile.

        @param y: array-like of wall-normal coordinates in [-1, 1]
        @return: array of streamwise velocities U(y)
        '''
        pass

    @abstractmethod
    def Uyy(self, y):
        '''
        Second wall-normal derivative U''(y).

        @param y: array-like of wall-normal coordinates in [-1, 1]
        @return: array of U''(y) values
        '''
        pass

    def __repr__(self):
        return f"{self.name} flow"

    def plot(self):
        import matplotlib.pyplot as plt

        y = np.linspace(-1, 1, 100)
        plt.plot(y, self.U(y), label='U(y)')
        plt.title(self.name)
        plt.xlabel('y')
        plt.ylabel('Flow Velocity')
        plt.legend()
        plt.grid()
        plt.show()


class PoiseuilleFlow(BaseFlow):
    '''
    Plane Poiseuille flow: U(y) = 1 - y²

    Pressure-driven flow between two stationary walls at y = ±1.
    Linear critical parameters: Re_c ≈ 5772,  α_c ≈ 1.02.
    '''

    @property
    def name(self):
        return "Plane Poiseuille"

    def U(self, y):
        return 1.0 - y**2

    def Uyy(self, y):
        return -2.0 * np.ones_like(np.asarray(y, dtype=float))


class CouetteFlow(BaseFlow):
    '''
    Plane Couette flow: U(y) = y

    Flow between a stationary wall (y = -1) and a moving wall (y = +1).
    Linearly stable for all Re (non-normal growth only).
    '''

    @property
    def name(self):
        return "Plane Couette"

    def U(self, y):
        return np.asarray(y, dtype=float).copy()

    def Uyy(self, y):
        return np.zeros_like(np.asarray(y, dtype=float))


class CustomFlow(BaseFlow):
    '''
    User-defined parallel flow profile.

    U''(y) is computed numerically via central differences if not provided.

    @param U_func:   callable U(y) — base flow velocity
    @param name:     display name for titles/labels (default "Custom")
    @param Uyy_func: callable U''(y) — if None, computed numerically
    '''

    def __init__(self, U_func, name="Custom", Uyy_func=None):
        self._U_func = U_func
        self._name = name
        self._Uyy_func = Uyy_func

    @property
    def name(self):
        return self._name

    def U(self, y):
        return np.asarray(self._U_func(np.asarray(y, dtype=float)), dtype=float)

    def Uyy(self, y):
        if self._Uyy_func is not None:
            return np.asarray(self._Uyy_func(np.asarray(y, dtype=float)), dtype=float)
        # Fourth-order central difference
        y = np.asarray(y, dtype=float)
        h = 1e-4
        return (-self._U_func(y + 2*h) + 16*self._U_func(y + h)
                - 30*self._U_func(y)
                + 16*self._U_func(y - h) - self._U_func(y - 2*h)) / (12 * h**2)
