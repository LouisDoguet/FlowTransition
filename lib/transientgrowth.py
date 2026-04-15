import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

from lib.orrsommerfeld import OrrSommerfeld


# ---------------------------------------------------------------------------
# Clenshaw-Curtis quadrature weights for Chebyshev collocation grid
# ---------------------------------------------------------------------------

def _clenshaw_curtis_weights(N):
    """
    Clenshaw-Curtis quadrature weights for the N Chebyshev collocation
    points  y_j = cos(j π / (N-1)),  j = 0 … N-1.

    The L² integral over [-1, 1] is approximated as  w^T f.
    Weights sum to 2 (the measure of [-1, 1]).

    Algorithm: DCT-mirror of the reciprocal-square frequency coefficients,
    followed by an inverse FFT (Waldvogel 2006).
    """
    n = N - 1
    if n == 0:
        return np.array([2.0])

    c = np.zeros(n + 1)
    c[0::2] = 2.0 / (1.0 - np.arange(0, n + 1, 2) ** 2)
    c_mirror = np.concatenate([c, c[n - 1:0:-1]])
    w = np.real(np.fft.ifft(c_mirror))
    w[0] /= 2.0
    w[n] /= 2.0
    return w[:n + 1]


# ---------------------------------------------------------------------------
# Transient growth
# ---------------------------------------------------------------------------

class TransientGrowth:
    """
    Maximum transient kinetic-energy growth for a parallel shear flow.

    Given M selected Orr-Sommerfeld eigenmodes (φ_j, c_j), the perturbation
    wall-normal velocity is decomposed as

        v(y, t) = Σ_j a_j φ_j(y) exp(σ_j t),   σ_j = i α c_j.

    The total kinetic energy at time t is measured by

        E(t) = a^H [exp(Λt)]^H Q exp(Λt) a

    where  Q = Ψ^H W Ψ  is the M×M kinetic-energy Gram matrix,
    Ψ (N×M) is the eigenfunction matrix, and W = diag(w_j) carries the
    Clenshaw-Curtis quadrature weights approximating the L² inner product.

    The maximum transient growth subject to unit initial energy is

        G(t) = max_{a : a^H Q a = 1}  a^H [exp(Λt)]^H Q exp(Λt) a

    Solving via SVD:  writing Q = F^H F (Cholesky) and substituting b = Fa,

        G(t) = σ_max( F exp(Λt) F^{-1} )²

    The optimal initial-condition coefficient vector is

        a_opt = F^{-1} v_1

    where v_1 is the first right singular vector of F exp(Λt) F^{-1}.

    Parameters
    ----------
    solver  : OrrSommerfeld
        A solved OrrSommerfeld instance (must have called .solve() first).
    n_modes : int, optional
        Number of eigenmodes to retain, taken in order of Im(c) descending
        (most unstable / least stable first).  Default: all available modes.

    Examples
    --------
    >>> from lib.baseflow import CouetteFlow
    >>> from lib.orrsommerfeld import OrrSommerfeld
    >>> from lib.transientgrowth import TransientGrowth
    >>> import numpy as np
    >>>
    >>> solver = OrrSommerfeld(CouetteFlow(), Re=1000, alpha=1.0).solve()
    >>> tg = TransientGrowth(solver, n_modes=50)
    >>> t_arr = np.linspace(0, 80, 200)
    >>> tg.plot_growth(t_arr)
    """

    def __init__(self, solver, n_modes=None):
        if not isinstance(solver, OrrSommerfeld):
            raise TypeError("solver must be an OrrSommerfeld instance.")
        if solver._eigenvalues is None:
            raise RuntimeError("Call solver.solve() before constructing TransientGrowth.")

        self._solver = solver
        M_total = len(solver.eigenvalues)

        if n_modes is None:
            n_modes = M_total
        if not (1 <= n_modes <= M_total):
            raise ValueError(
                f"n_modes must be between 1 and {M_total}; got {n_modes}."
            )
        self.n_modes = n_modes

        # Pre-build the energy matrix and its Cholesky factor once
        self._Q, self._F, self._F_inv = self._build_energy_matrix()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_energy_matrix(self):
        """
        Build the M×M kinetic-energy Gram matrix Q = Ψ^H W Ψ and its
        Cholesky factorisation Q = F^H F (F upper triangular).

        W = diag(w_j) carries the Clenshaw-Curtis quadrature weights so that
        the discrete inner product  ⟨f, g⟩ = f^H W g  approximates the L²
        integral over [-1, 1].

        Returns
        -------
        Q     : complex ndarray, shape (M, M)
        F     : complex ndarray, shape (M, M)  — upper-triangular Cholesky factor
        F_inv : complex ndarray, shape (M, M)  — inverse of F
        """
        Psi = self._solver.eigenvectors[:, :self.n_modes]   # N × M
        N = Psi.shape[0]

        w = _clenshaw_curtis_weights(N)             # Clenshaw-Curtis weights
        W_sqrt = np.sqrt(w)[:, np.newaxis]          # (N, 1) for broadcasting

        # U_eff = W^{1/2} Ψ  →  Q = U_eff^H U_eff = Ψ^H W Ψ
        U_eff = W_sqrt * Psi                        # N × M
        Q = U_eff.conj().T @ U_eff                  # M × M, Hermitian PD

        # Symmetrize to suppress floating-point asymmetry
        Q = (Q + Q.conj().T) * 0.5

        # Cholesky: Q = F^H F  (upper=True → F is upper-triangular)
        F = cholesky(Q, lower=False)
        F_inv = np.linalg.inv(F)

        return Q, F, F_inv

    def _evolution_diagonal(self, t):
        """
        Return the diagonal entries of exp(Λt):  exp(λ_j t),  j = 0 … M-1.

        The mode exp(i(αx − ωt)) has temporal eigenvalue  λ_j = −iω_j = −iαc_j.
        For a stable mode Im(c_j) < 0, so  Re(λ_j) = α Im(c_j) < 0  (decays). ✓
        """
        c = self._solver.eigenvalues[:self.n_modes]
        lam = -1j * self._solver.alpha * c      # temporal eigenvalues λ_j = −iαc_j
        return np.exp(lam * t)

    def _svd_at_t(self, t):
        """
        Build  M_mat = F exp(Λt) F^{-1}  and return its SVD.

        Returns (U, s, Vh) from np.linalg.svd.
        """
        exp_vals = self._evolution_diagonal(t)
        M_mat = self._F @ np.diag(exp_vals) @ self._F_inv
        return np.linalg.svd(M_mat)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def Q(self):
        """Kinetic-energy Gram matrix Q = Ψ^H W Ψ, shape (n_modes, n_modes)."""
        return self._Q

    def compute(self, t):
        """
        Compute the maximum transient growth G(t).

        G(t) = σ_max( F exp(Λt) F^{-1} )²

        Parameters
        ----------
        t : float or array-like
            Time(s) at which to evaluate G.

        Returns
        -------
        G : float or ndarray
            Maximum transient growth.  Scalar for scalar input; array
            with the same shape as t otherwise.
        """
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        G = np.empty(t.shape)
        for i, ti in enumerate(t):
            _, s, _ = self._svd_at_t(ti)
            G[i] = s[0] ** 2

        return float(G[0]) if scalar else G

    def optimal_initial_condition(self, t):
        """
        Return the optimal initial perturbation that maximises G(t).

        The profile is normalised to unit initial kinetic energy:
            a_opt^H Q a_opt = 1.

        Parameters
        ----------
        t : float
            Target time.

        Returns
        -------
        y     : ndarray, shape (N,)
            Chebyshev collocation grid.
        v_opt : complex ndarray, shape (N,)
            Wall-normal velocity profile of the optimal perturbation.
        a_opt : complex ndarray, shape (n_modes,)
            Modal coefficient vector (unit energy: a_opt^H Q a_opt = 1).
        G_t   : float
            Transient growth G(t) achieved by v_opt.
        """
        t = float(t)
        _, s, Vh = self._svd_at_t(t)
        G_t = float(s[0] ** 2)

        # First right singular vector: Vh stores V^H, so v1 = Vh[0].conj()
        a_opt = self._F_inv @ Vh[0].conj()

        Psi = self._solver.eigenvectors[:, :self.n_modes]
        v_opt = Psi @ a_opt

        return self._solver.y, v_opt, a_opt, G_t

    def peak_growth(self, t_array):
        """
        Find the peak transient growth over the given time array.

        Parameters
        ----------
        t_array : array-like
            Time values to sweep.

        Returns
        -------
        G_max : float
            Maximum value of G(t) over t_array.
        t_max : float
            Time at which G_max is attained.
        """
        t_arr = np.asarray(t_array, dtype=float)
        G_arr = self.compute(t_arr)
        idx = int(np.argmax(G_arr))
        return float(G_arr[idx]), float(t_arr[idx])

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_growth(self, t_array, ax=None, title=None, figsize=(8, 5),
                    log_scale=False, mark_peak=True):
        """
        Plot G(t) versus time.

        Parameters
        ----------
        t_array   : array-like  — time values to sweep
        ax        : Axes, opt   — existing axes (creates figure if None)
        title     : str, opt
        figsize   : tuple
        log_scale : bool        — logarithmic y-axis (default False)
        mark_peak : bool        — annotate the peak (default True)
        """
        t_arr = np.asarray(t_array, dtype=float)
        G_arr = self.compute(t_arr)
        G_max, t_max = float(G_arr.max()), float(t_arr[G_arr.argmax()])

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(t_arr, G_arr, color='steelblue', linewidth=2)
        ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5,
                   label='$G = 1$')

        if mark_peak:
            ax.axvline(t_max, color='tomato', linewidth=0.9, linestyle=':',
                       alpha=0.7)
            ax.scatter([t_max], [G_max], color='tomato', zorder=5,
                       label=f'$G_{{max}} = {G_max:.3f}$  at  $t = {t_max:.2f}$')
            ax.legend(fontsize=10)

        ax.set_xlabel(r'$t$', fontsize=13)
        ax.set_ylabel(r'$G(t)$', fontsize=13)

        if log_scale:
            ax.set_yscale('log')

        if title is None:
            title = (
                f"{self._solver.flow.name} — Max Transient Growth\n"
                f"Re = {self._solver.Re},  "
                f"α = {self._solver.alpha},  "
                f"β = {self._solver.beta},  "
                f"M = {self.n_modes} modes"
            )
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.25)

        if standalone:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_optimal_mode(self, t, title=None, figsize=(10, 5)):
        """
        Plot the amplitude and phase of the optimal initial perturbation v_opt(y).

        Parameters
        ----------
        t      : float  — target time
        title  : str, optional
        figsize: tuple
        """
        y, v_opt, _, G_t = self.optimal_initial_condition(t)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

        ax1.plot(np.abs(v_opt), y, color='royalblue', linewidth=2)
        ax1.set_xlabel(r'$|\hat{v}_{opt}(y)|$', fontsize=13)
        ax1.set_ylabel(r'$y$', fontsize=13)
        ax1.set_xlim(left=0)
        ax1.grid(True, alpha=0.25)

        ax2.plot(np.angle(v_opt), y, color='tomato', linewidth=2)
        ax2.set_xlabel(r'$\arg(\hat{v}_{opt}(y))$', fontsize=13)
        ax2.set_xlim([-np.pi, np.pi])
        ax2.grid(True, alpha=0.25)

        if title is None:
            title = (
                f"{self._solver.flow.name} — Optimal Initial Perturbation\n"
                f"$t = {t}$,  $G(t) = {G_t:.4f}$,  "
                f"Re = {self._solver.Re},  "
                f"α = {self._solver.alpha},  "
                f"M = {self.n_modes} modes"
            )

        fig.suptitle(title, fontsize=11)
        plt.tight_layout()
        plt.show()

        return fig

    def plot_spectrum_and_growth(self, t_array, figsize=(14, 5)):
        """
        Side-by-side plot: eigenspectrum (left) and G(t) curve (right).

        Parameters
        ----------
        t_array : array-like  — time values for the growth curve
        figsize : tuple
        """
        fig, (ax_spec, ax_growth) = plt.subplots(1, 2, figsize=figsize)

        self._solver.plot_spectrum(ax=ax_spec, colorbar=False)
        self.plot_growth(t_array, ax=ax_growth)

        fig.suptitle(
            f"{self._solver.flow.name}   Re = {self._solver.Re},  "
            f"α = {self._solver.alpha},  β = {self._solver.beta}",
            fontsize=13
        )
        plt.tight_layout()
        plt.show()

        return fig

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"TransientGrowth("
            f"flow={self._solver.flow.name!r}, "
            f"Re={self._solver.Re}, "
            f"alpha={self._solver.alpha}, "
            f"beta={self._solver.beta}, "
            f"n_modes={self.n_modes})"
        )
