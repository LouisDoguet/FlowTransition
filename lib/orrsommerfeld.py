import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.linalg import eig

from lib.baseflow import BaseFlow


# ---------------------------------------------------------------------------
# Chebyshev spectral utilities
# ---------------------------------------------------------------------------

def _cheb(N):
    '''
    Build an N×N Chebyshev differentiation matrix on [-1, 1].

    Grid points: y_j = cos(j π / (N-1)),  j = 0 … N-1
    So y_0 = 1 (top wall) and y_{N-1} = -1 (bottom wall).

    @param N: total number of grid points (including both boundary points)
    @return: (D, y)  —  D is N×N,  y is length-N array
    '''
    if N == 1:
        return np.array([[0.0]]), np.array([1.0])

    j = np.arange(N)
    y = np.cos(np.pi * j / (N - 1))

    # Chebyshev weights
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0)**j

    # np.tile(y, (N, 1)) creates Y[i,j] = y[j]  (rows are copies of y).
    # Y.T[i,j] = y[i], so  Y.T - Y  gives  dY[i,j] = y[i] - y[j],
    # matching Trefethen's formula D[i,j] = (c_i/c_j) / (y_i - y_j).
    # Using  Y - Y.T  (the reverse) would give -d/dξ instead of +d/dξ.
    Y = np.tile(y, (N, 1))
    dY = Y.T - Y                          # dY[i,j] = y[i] - y[j]  ← correct sign

    # Avoid division by zero on the diagonal; we'll overwrite it
    np.fill_diagonal(dY, 1.0)

    D = np.outer(c, 1.0 / c) / dY

    # Diagonal by the negative-sum rule  D_ii = -∑_{j≠i} D_ij
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))

    return D, y


# ---------------------------------------------------------------------------
# Orr-Sommerfeld solver
# ---------------------------------------------------------------------------

class OrrSommerfeld:
    '''
    Orr-Sommerfeld eigenvalue solver for parallel shear flows.

    Solves the Orr-Sommerfeld equation using Chebyshev spectral collocation:

        [(U - c)(D² - k²) - U''] φ = (1 / iαRe) (D² - k²)² φ

    Eigenvalue: complex phase velocity  c = ω/α = c_r + i c_i
        c_i > 0  →  temporally unstable (growing) mode
        c_i < 0  →  temporally stable   (decaying) mode

    Wavenumbers: k² = α² + β²  (β = 0 for 2-D perturbations)

    Boundary conditions: φ = Dφ = 0  at  y = ±1  (no-slip / no-penetration)

    Parameters
    ----------
    flow  : BaseFlow
        Base flow profile (PoiseuilleFlow, CouetteFlow, CustomFlow, …).
    Re    : float
        Reynolds number.
    alpha : float
        Streamwise wavenumber α.
    beta  : float, optional
        Spanwise wavenumber β (default 0 → 2-D perturbation).
    N     : int, optional
        Number of Chebyshev collocation points including both walls
        (default 128).  N ≥ 64 recommended for well-resolved branches.
    '''

    def __init__(self, flow, Re, alpha, beta=0.0, N=128):
        if not isinstance(flow, BaseFlow):
            raise TypeError("flow must be a BaseFlow instance (PoiseuilleFlow, CouetteFlow, …)")

        self.flow = flow
        self.Re = Re
        self.alpha = alpha
        self.beta = beta
        self.N = N

        self._eigenvalues = None
        self._eigenvectors = None
        self._y = None

    # ------------------------------------------------------------------
    # Core solver
    # ------------------------------------------------------------------

    def solve(self):
        '''
        Solve the Orr-Sommerfeld eigenvalue problem.

        Populates self.eigenvalues (sorted by Im(c) descending — most
        unstable first) and self.eigenvectors.

        Returns self so calls can be chained:  solver.solve().plot_spectrum()
        '''
        N = self.N
        alpha = self.alpha
        beta = self.beta
        Re = self.Re
        k2 = alpha**2 + beta**2

        D, y = _cheb(N)
        self._y = y

        I = np.eye(N)
        D2 = D @ D
        D4 = D2 @ D2

        U_vec = self.flow.U(y)
        Uyy_vec = self.flow.Uyy(y)

        U_diag = np.diag(U_vec)
        Uyy_diag = np.diag(Uyy_vec)

        # ---- Build OS operators ----------------------------------------
        # Temporal eigenvalue problem: A φ = c B φ
        #
        # From (U - c)(D²-k²)φ - U''φ = (1/iαRe)(D²-k²)²φ
        # →  c (D²-k²)φ = [U(D²-k²) - U'' - (1/iαRe)(D²-k²)²] φ
        #
        #   A = U(D²-k²) - U'' - (1/iαRe)(D²-k²)²
        #   B = D² - k²
        #
        L = D2 - k2 * I                           # (D² - k²)

        A = U_diag @ L - Uyy_diag - (1.0 / (1j * alpha * Re)) * (L @ L)
        B = L.copy()

        # ---- Apply boundary conditions ---------------------------------
        # Four BCs:  φ(1) = 0,  φ'(1) = 0,  φ'(-1) = 0,  φ(-1) = 0
        # Implemented by replacing rows 0, 1, N-2, N-1 in A.
        # Zeroing matching rows in B makes those eigenvalues → ∞ (filtered later).

        for row in [0, 1, N - 2, N - 1]:
            B[row, :] = 0.0

        # φ(1) = 0   →  row 0: e_0 · φ = 0
        A[0, :] = 0.0
        A[0, 0] = 1.0

        # φ'(1) = 0  →  row 1: D[0, :] · φ = 0  (D[0,:] evaluates d/dy at y=1)
        A[1, :] = D[0, :]

        # φ'(-1) = 0 →  row N-2: D[-1, :] · φ = 0  (D[-1,:] evaluates d/dy at y=-1)
        A[N - 2, :] = D[-1, :]

        # φ(-1) = 0  →  row N-1: e_{N-1} · φ = 0
        A[N - 1, :] = 0.0
        A[N - 1, N - 1] = 1.0

        # ---- Solve generalized eigenvalue problem ----------------------
        eigenvalues, eigenvectors = eig(A, B)

        # ---- Filter spurious eigenvalues from BC rows -----------------
        finite_mask = (np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 1e6))
        eigenvalues = eigenvalues[finite_mask]
        eigenvectors = eigenvectors[:, finite_mask]

        # Sort by Im(c) descending: most unstable first
        sort_idx = np.argsort(-eigenvalues.imag)
        self._eigenvalues = eigenvalues[sort_idx]
        self._eigenvectors = eigenvectors[:, sort_idx]

        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def eigenvalues(self):
        '''Complex phase velocities c = c_r + i c_i, most unstable first.'''
        if self._eigenvalues is None:
            raise RuntimeError("Call .solve() before accessing eigenvalues.")
        return self._eigenvalues

    @property
    def eigenvectors(self):
        '''Eigenfunctions φ(y), columns aligned with self.eigenvalues.'''
        if self._eigenvectors is None:
            raise RuntimeError("Call .solve() before accessing eigenvectors.")
        return self._eigenvectors

    @property
    def y(self):
        '''Chebyshev collocation grid points on [-1, 1].'''
        if self._y is None:
            raise RuntimeError("Call .solve() before accessing y.")
        return self._y

    @property
    def growth_rate(self):
        '''Maximum temporal growth rate  ω_i = α · max(Im(c)).'''
        return self.alpha * float(self.eigenvalues[0].imag)

    @property
    def is_unstable(self):
        '''True if at least one eigenvalue has Im(c) > 0.'''
        return bool(self.eigenvalues[0].imag > 0)

    # ------------------------------------------------------------------
    # Visualisation: eigenspectrum
    # ------------------------------------------------------------------

    def plot_spectrum(self, title=None, figsize=(8, 7),
                      colorbar=True, mark_unstable=True, ax=None,
                      cr_lim=None, ci_lim=None):
        '''
        Plot the eigenspectrum in the complex c-plane (c_r, c_i).

        Eigenvalues are colour-coded by c_r so the three branches of the
        classical Y-shaped spectrum (A, P, S) are visually distinct:
          — A branch  (wall modes)    :  c_r ≈ U_wall  (blue end)
          — P branch  (centre modes)  :  c_r ≈ U_max   (red end)
          — S branch  (connecting)    :  intermediate c_r (green)

        Only eigenvalues within the requested c_r / c_i window are plotted,
        discarding the numerically spurious modes outside the physical range.

        Parameters
        ----------
        title        : str, optional        —  custom plot title
        figsize      : tuple                —  figure size
        colorbar     : bool                 —  show c_r colour bar
        mark_unstable: bool                 —  highlight unstable modes (c_i > 0)
        ax           : Axes, optional       —  existing axes to plot into
        cr_lim       : (float, float), opt  —  (c_r_min, c_r_max) window
                       defaults to (U_wall - 0.05, U_max + 0.05)
        ci_lim       : (float, float), opt  —  (c_i_min, c_i_max) window
                       defaults to (-0.7, 0.05)
        '''
        if self._eigenvalues is None:
            raise RuntimeError("Call .solve() first.")

        c_all = self._eigenvalues
        y = self._y

        U_vals = self.flow.U(y)
        U_min, U_max = U_vals.min(), U_vals.max()

        # ---- default axis windows ------------------------------------
        if cr_lim is None:
            cr_lim = (U_min - 0.05, U_max + 0.05)
        if ci_lim is None:
            ci_lim = (-0.7, 0.05)

        # ---- keep only eigenvalues inside the physical window --------
        mask = ((c_all.real >= cr_lim[0]) & (c_all.real <= cr_lim[1]) &
                (c_all.imag >= ci_lim[0]) & (c_all.imag <= ci_lim[1]))
        c = c_all[mask]

        standalone = (ax is None)
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)

        # ---- colour map over c_r range --------------------------------
        norm = Normalize(vmin=U_min, vmax=U_max)
        cmap = plt.cm.RdYlBu_r

        sc = ax.scatter(c.real, c.imag,
                        c=c.real, cmap=cmap, norm=norm,
                        s=14, alpha=0.85, zorder=3)

        if colorbar and standalone:
            cb = plt.colorbar(sc, ax=ax, pad=0.02)
            cb.set_label(r'$c_r$', fontsize=12)

        # ---- highlight unstable modes ---------------------------------
        if mark_unstable:
            unstable = c[c.imag > 0]
            if len(unstable) > 0:
                ax.scatter(unstable.real, unstable.imag,
                           s=120, marker='*', color='limegreen',
                           edgecolors='black', linewidths=0.5,
                           zorder=6, label=f'Unstable  ({len(unstable)})')
                ax.legend(fontsize=10)

        # ---- reference lines -----------------------------------------
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)
        ax.axvline(U_min, color='steelblue', linewidth=0.7,
                   linestyle=':', alpha=0.5, label=f'$U_{{wall}}={U_min:.2f}$')
        ax.axvline(U_max, color='tomato', linewidth=0.7,
                   linestyle=':', alpha=0.5, label=f'$U_{{max}}={U_max:.2f}$')

        # ---- enforce axis limits -------------------------------------
        ax.set_xlim(cr_lim)
        ax.set_ylim(ci_lim)

        if title is None:
            title = (f"{self.flow.name} — Eigenspectrum\n"
                     f"Re = {self.Re},  α = {self.alpha},  "
                     f"β = {self.beta},  N = {self.N}")

        ax.set_xlabel(r'$c_r = \mathrm{Re}(c)$', fontsize=12)
        ax.set_ylabel(r'$c_i = \mathrm{Im}(c)$', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.25)

        if standalone:
            plt.tight_layout()
            plt.show()

        return ax

    # ------------------------------------------------------------------
    # Squire equation solver (wall-normal vorticity forced by OS mode)
    # ------------------------------------------------------------------

    def _solve_squire(self, c, phi):
        '''
        Solve the Squire equation for the wall-normal vorticity η̂(y),
        forced by the OS eigenfunction φ = v̂.

        Squire equation:
            (U - c) η̂ - (1/iαRe)(D² - k²) η̂ = iβ U' v̂

        BCs: η̂(±1) = 0  (no-slip at walls)

        Returns η̂ as a complex array on the same Chebyshev grid.
        For β = 0 the forcing vanishes and η̂ = 0 exactly.
        '''
        if self.beta == 0.0:
            return np.zeros_like(phi)

        N = self.N
        D, y = _cheb(N)
        I = np.eye(N)
        D2 = D @ D
        k2 = self.alpha**2 + self.beta**2

        U_vec = self.flow.U(y)
        Uy_vec = D @ U_vec          # U'(y) via Chebyshev differentiation

        L = D2 - k2 * I

        # System matrix
        M = np.diag(U_vec - c) - (1.0 / (1j * self.alpha * self.Re)) * L

        # Forcing
        f = 1j * self.beta * Uy_vec * phi

        # Enforce η̂(±1) = 0
        for row in [0, N - 1]:
            M[row, :] = 0.0
            M[row, row] = 1.0
            f[row] = 0.0

        return np.linalg.solve(M, f)

    # ------------------------------------------------------------------
    # Visualisation: eigenfunction
    # ------------------------------------------------------------------

    def plot_eigenmode(self, mode_index=0, title=None, figsize=(12, 6)):
        '''
        Plot |v̂(y)| and |η̂(y)| — the amplitude profiles of the
        wall-normal velocity and wall-normal vorticity eigenfunctions.

        v̂ comes directly from the OS eigenvalue problem.
        η̂ is obtained by solving the forced Squire equation for the
        same eigenvalue c.  For β = 0, η̂ = 0 identically.

        Parameters
        ----------
        mode_index : int  —  0 = most unstable / least stable (default)
        title      : str, optional
        figsize    : tuple
        '''
        if self._eigenvalues is None:
            raise RuntimeError("Call .solve() first.")

        c = self._eigenvalues[mode_index]
        phi = self._eigenvectors[:, mode_index].copy()
        y = self._y

        # Normalise v̂ so its peak amplitude is 1
        phi /= np.max(np.abs(phi))

        # Solve for η̂
        eta = self._solve_squire(c, phi)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

        ax1.plot(np.abs(phi), y, color='royalblue', linewidth=2)
        ax1.set_xlabel(r'$|\tilde{v}(y)|$', fontsize=13)
        ax1.set_ylabel(r'$y$', fontsize=13)
        ax1.set_xlim(left=0)
        ax1.grid(True, alpha=0.25)

        ax2.plot(np.abs(eta), y, color='tomato', linewidth=2)
        ax2.set_xlabel(r'$|\tilde{\eta}(y)|$', fontsize=13)
        ax2.set_xlim(left=0)
        ax2.grid(True, alpha=0.25)

        if title is None:
            stability_tag = "unstable" if c.imag > 0 else "stable"
            title = (f"{self.flow.name} — Mode #{mode_index}  ({stability_tag})\n"
                     f"$c = {c.real:.5f} + {c.imag:.5f}\\,i$,  "
                     f"Re = {self.Re},  α = {self.alpha},  β = {self.beta}")

        fig.suptitle(title, fontsize=11)
        plt.tight_layout()
        plt.show()

        return fig

    # ------------------------------------------------------------------
    # Visualisation: multi-Re eigenspectrum
    # ------------------------------------------------------------------

    def plot_spectrum_grid(self, Re_values, alpha=None, beta=None, N=None,
                           figsize=None, cols=3):
        '''
        Plot eigenspectra for a list of Reynolds numbers in a grid of subplots.

        Useful for visualising how the Y-shaped branches evolve with Re and
        watching the Tollmien-Schlichting mode cross into the unstable half-plane.

        Parameters
        ----------
        Re_values : list of float  —  Reynolds numbers to sweep
        alpha     : float, optional  —  overrides self.alpha for all panels
        beta      : float, optional  —  overrides self.beta
        N         : int,   optional  —  overrides self.N
        figsize   : tuple, optional
        cols      : int              —  number of subplot columns (default 3)
        '''
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        N = self.N if N is None else N

        n = len(Re_values)
        rows = int(np.ceil(n / cols))
        if figsize is None:
            figsize = (5 * cols, 4.5 * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).flatten()

        for i, Re_i in enumerate(Re_values):
            solver_i = OrrSommerfeld(self.flow, Re_i, alpha, beta, N)
            solver_i.solve()
            solver_i.plot_spectrum(
                title=f"Re = {Re_i}",
                colorbar=False,
                ax=axes[i]
            )

        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        suptitle = (f"{self.flow.name} — Eigenspectrum vs Re\n"
                    f"α = {alpha},  β = {beta}")
        fig.suptitle(suptitle, fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

        return fig
