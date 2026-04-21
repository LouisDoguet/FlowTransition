import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.integrate import solve_ivp

from lib.orrsommerfeld import _cheb


# ---------------------------------------------------------------------------
# Chebyshev differentiation matrix on an arbitrary interval [a, b]
# ---------------------------------------------------------------------------

def _cheb_interval(N, a, b):
    '''
    Chebyshev differentiation matrix on [a, b].

    Remaps the standard Chebyshev grid on [-1, 1] to [a, b] via the linear
    change of variables  x = (b - a)/2 · ξ + (a + b)/2.

    @param N: number of collocation points (including both endpoints)
    @param a: left endpoint
    @param b: right endpoint
    @return: (D, x) — D is N×N derivative matrix on [a, b]; x is the grid
    '''
    D_std, xi = _cheb(N)
    D = D_std * (2.0 / (b - a))
    x = (b - a) / 2.0 * xi + (a + b) / 2.0
    return D, x


# ---------------------------------------------------------------------------
# Forcing terms  f(x, t)
# ---------------------------------------------------------------------------

class Forcing(ABC):
    '''
    Abstract base class for the forcing term f(x, t) in the Ginzburg-Landau
    equation:

        ∂A/∂t = μ(x) A + γ ∂²A/∂x² − U_g ∂A/∂x + f(x, t)

    Subclasses must implement __call__(x, t) and the name property.
    '''

    @property
    @abstractmethod
    def name(self):
        '''Human-readable name of the forcing type.'''
        pass

    @abstractmethod
    def __call__(self, x, t):
        '''
        Evaluate the forcing at spatial positions x and time t.

        @param x: ndarray of spatial positions
        @param t: float, current time
        @return: complex ndarray, same shape as x
        '''
        pass

    def __repr__(self):
        return self.name

    def plot(self, x, t_values=None, figsize=(9, 4)):
        '''
        Plot |f(x, t)| for a set of time snapshots.

        @param x:        array-like, spatial positions
        @param t_values: list of floats, times to evaluate (default [0])
        @param figsize:  tuple, figure size
        '''
        if t_values is None:
            t_values = [0.0]
        x = np.asarray(x, dtype=float)

        fig, ax = plt.subplots(figsize=figsize)
        for t in t_values:
            ax.plot(x, np.abs(self(x, float(t))), label=f't = {t:.3g}')

        ax.set_xlabel(r'$x$', fontsize=13)
        ax.set_ylabel(r'$|f(x,\,t)|$', fontsize=12)
        ax.set_title(self.name, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.show()
        return fig


class PointForcing(Forcing):
    '''
    Spatially localised source with arbitrary temporal modulation, approximating
    a Dirac delta at x₀ with a normalised Gaussian of width σ:

        f(x, t) = A0 · g(t) · G_σ(x − x₀)

    As σ → 0 this converges weakly to  A0 · g(t) · δ(x − x₀).
    Set g = lambda t: 1 for a steady point source.

    @param x0:    float, forcing location
    @param g:     callable g(t) → scalar (real or complex), temporal envelope.
                  Default: constant 1 (steady).
    @param A0:    float or complex, amplitude (default 1.0)
    @param sigma: float, Gaussian half-width (default 0.5). Should be a few
                  times the grid spacing to avoid spatial aliasing.
    '''

    def __init__(self, x0, g=None, A0=1.0, sigma=0.5):
        self.x0 = float(x0)
        self._g = g if g is not None else (lambda t: 1.0)
        self.A0 = complex(A0)
        self.sigma = float(sigma)

    @property
    def name(self):
        return f"Point source at x₀ = {self.x0}"

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        gaussian = np.exp(-0.5 * ((x - self.x0) / self.sigma) ** 2)
        gaussian /= self.sigma * np.sqrt(2.0 * np.pi)
        return self.A0 * self._g(float(t)) * gaussian


class HarmonicForcing(Forcing):
    '''
    Monochromatic harmonic point source oscillating at angular frequency ω₀:

        f(x, t) = A0 · exp(i ω₀ t) · G_σ(x − x₀)

    Models a vibrating ribbon in a linear stability experiment.
    The long-time response reveals the spatial Green's function of the GL
    operator at frequency ω₀.

    @param x0:    float, forcing location
    @param omega: float, forcing angular frequency ω₀
    @param A0:    float or complex, amplitude (default 1.0)
    @param sigma: float, Gaussian width for spatial localisation (default 0.5)
    '''

    def __init__(self, x0, omega, A0=1.0, sigma=0.5):
        self.x0 = float(x0)
        self.omega = float(omega)
        self.A0 = complex(A0)
        self.sigma = float(sigma)

    @property
    def name(self):
        return f"Harmonic source (ω₀ = {self.omega:.4g}) at x₀ = {self.x0}"

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        gaussian = np.exp(-0.5 * ((x - self.x0) / self.sigma) ** 2)
        gaussian /= self.sigma * np.sqrt(2.0 * np.pi)
        return self.A0 * np.exp(1j * self.omega * float(t)) * gaussian


class ImpulseForcing(Forcing):
    '''
    Spatio-temporal impulse: narrow Gaussian in both x and t, approximating a
    Dirac delta in space and time simultaneously:

        f(x, t) = A0 · G_{σ_x}(x − x₀) · G_{σ_t}(t − t₀)

    Excites the causal Green's function (impulse response) of the GL operator.
    Use σ_x ≫ Δx and σ_t ≫ Δt to keep the pulse resolved on the grid.

    @param x0:      float, spatial location of the impulse
    @param t0:      float, time of the impulse
    @param A0:      float or complex, amplitude (default 1.0)
    @param sigma_x: float, spatial Gaussian half-width (default 0.5)
    @param sigma_t: float, temporal Gaussian half-width (default 0.1)
    '''

    def __init__(self, x0, t0, A0=1.0, sigma_x=0.5, sigma_t=0.1):
        self.x0 = float(x0)
        self.t0 = float(t0)
        self.A0 = complex(A0)
        self.sigma_x = float(sigma_x)
        self.sigma_t = float(sigma_t)

    @property
    def name(self):
        return f"Impulse at (x₀ = {self.x0}, t₀ = {self.t0})"

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        g_x = np.exp(-0.5 * ((x - self.x0) / self.sigma_x) ** 2)
        g_x /= self.sigma_x * np.sqrt(2.0 * np.pi)
        g_t = np.exp(-0.5 * ((float(t) - self.t0) / self.sigma_t) ** 2)
        g_t /= self.sigma_t * np.sqrt(2.0 * np.pi)
        return self.A0 * g_x * g_t


class WavepacketForcing(Forcing):
    '''
    Localised wavepacket: Gaussian spatial envelope modulated by a carrier
    wavenumber k₀ and an arbitrary temporal gate g(t):

        f(x, t) = A0 · g(t) · exp(−(x − x₀)² / 2σ²) · exp(i k₀ (x − x₀))

    Launches a coherent wavepacket centred at x₀.  Setting g(t) = 1 gives a
    sustained monochromatic source; using g with finite support creates a pulse.

    @param x0:    float, centre of the wavepacket
    @param k0:    float, carrier wavenumber
    @param sigma: float, spatial envelope half-width
    @param g:     callable g(t) → scalar, temporal modulation (default: 1)
    @param A0:    float or complex, amplitude (default 1.0)
    '''

    def __init__(self, x0, k0, sigma, g=None, A0=1.0):
        self.x0 = float(x0)
        self.k0 = float(k0)
        self.sigma = float(sigma)
        self._g = g if g is not None else (lambda t: 1.0)
        self.A0 = complex(A0)

    @property
    def name(self):
        return (f"Wavepacket (k₀ = {self.k0:.4g}, σ = {self.sigma:.4g}) "
                f"at x₀ = {self.x0}")

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        envelope = np.exp(-0.5 * ((x - self.x0) / self.sigma) ** 2)
        carrier = np.exp(1j * self.k0 * (x - self.x0))
        return self.A0 * self._g(float(t)) * envelope * carrier


class CompositeForcing(Forcing):
    '''
    Linear superposition of multiple forcing terms: f = f₁ + f₂ + …

    @param forcings: two or more Forcing instances to superpose
    '''

    def __init__(self, *forcings):
        if len(forcings) < 1:
            raise ValueError("CompositeForcing needs at least one Forcing.")
        for fi in forcings:
            if not isinstance(fi, Forcing):
                raise TypeError(
                    f"All arguments must be Forcing instances; got {type(fi)}."
                )
        self._forcings = list(forcings)

    @property
    def name(self):
        return "Composite: " + " + ".join(fi.name for fi in self._forcings)

    def __call__(self, x, t):
        return sum(fi(x, t) for fi in self._forcings)

    def add(self, forcing):
        '''
        Append another Forcing to the superposition.

        @param forcing: Forcing instance
        @return: self (for chaining)
        '''
        if not isinstance(forcing, Forcing):
            raise TypeError(
                f"Expected a Forcing instance; got {type(forcing)}."
            )
        self._forcings.append(forcing)
        return self


# ---------------------------------------------------------------------------
# Ginzburg-Landau solver
# ---------------------------------------------------------------------------

class GinzburgLandau:
    '''
    Solver for the linearised Ginzburg-Landau (GL) equation on a bounded domain.

    The equation is:

        ∂A/∂t = μ(x) A + γ ∂²A/∂x² − U_g ∂A/∂x + f(x, t)

    where:
        A(x, t)  complex perturbation amplitude (the field we solve for)
        μ(x)     local complex growth rate; Re(μ) > 0 → locally unstable
        γ        complex diffusion / dispersion coefficient
        U_g      real group velocity (linear advection speed)
        f(x, t)  external forcing term (a Forcing instance or None)

    Spatial discretisation: Chebyshev spectral collocation on [x_min, x_max],
    reusing the _cheb() infrastructure from orrsommerfeld.py.

    Time integration: exponential ETD1 integrator by default.  The spatial
    operator L is diagonalised once (O(N³)); each time step is then O(N²)
    and unconditionally stable — no stiffness constraint.  The stiffness of
    Chebyshev operators scales as O(N⁴), making explicit Runge-Kutta methods
    (RK45) completely impractical for N ≳ 32.  Always use method='exponential'
    (default) or, as a fallback, method='BDF'.

    Boundary conditions: Dirichlet  A(x_min, t) = A(x_max, t) = 0.

    Parameters
    ----------
    mu    : float, complex, or callable
        Local growth rate.  A scalar gives a spatially uniform μ.
        A callable mu(x) allows any spatial variation, e.g.
            mu = lambda x: -0.3 + 0.1 * (1 - (x / 10)**2)
        for a parabolic profile that is unstable near x = 0.
    gamma : complex, optional
        Complex diffusion coefficient (default: 1 + 0j).
        Re(γ) > 0 is required for well-posedness.
    U_g   : float, optional
        Group velocity — advects the wavepacket downstream (default: 0).
    x_lim : (float, float)
        Spatial domain [x_min, x_max].  Default: (-20, 20).
    N     : int
        Number of Chebyshev collocation points including both walls.
        Default: 128.

    Examples
    --------
    >>> from lib.ginzburglandau import GinzburgLandau, HarmonicForcing
    >>> import numpy as np
    >>>
    >>> mu_func = lambda x: -0.3 + 0.1 * (1.0 - (x / 10.0)**2)
    >>> gl = GinzburgLandau(mu=mu_func, gamma=1+0.2j, U_g=2.0,
    ...                     x_lim=(-20, 20), N=128)
    >>> forcing = HarmonicForcing(x0=0.0, omega=0.5)
    >>> t_eval = np.linspace(0, 80, 400)
    >>> gl.solve(t_span=(0, 80), forcing=forcing, t_eval=t_eval)
    >>> gl.plot_spacetime()
    '''

    def __init__(self, mu, gamma=1.0+0j, U_g=0.0, x_lim=(-20.0, 20.0), N=128):
        self.mu = mu
        self.gamma = complex(gamma)
        self.U_g = float(U_g)
        self.x_lim = (float(x_lim[0]), float(x_lim[1]))
        self.N = int(N)

        if self.x_lim[0] >= self.x_lim[1]:
            raise ValueError("x_lim[0] must be strictly less than x_lim[1].")
        if self.N < 4:
            raise ValueError("N must be at least 4.")

        # Build spatial operators — computed once in the constructor
        self._D, self._x = _cheb_interval(N, self.x_lim[0], self.x_lim[1])
        self._D2 = self._D @ self._D
        self._L = self._build_operator()

        # Solution storage (populated by solve())
        self._t = None
        self._A = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_operator(self):
        '''
        Assemble the N×N spatial operator  L  such that  ∂A/∂t = L A + f.

        L = diag(μ(x)) + γ D² − U_g D

        Dirichlet BCs are imposed by zeroing rows 0 and N-1 of L so that
        the boundary nodes are held at zero throughout the integration.
        '''
        N = self.N
        x = self._x

        if callable(self.mu):
            mu_vec = np.asarray(self.mu(x), dtype=complex)
        else:
            mu_vec = np.full(N, complex(self.mu), dtype=complex)

        L = (np.diag(mu_vec)
             + self.gamma * self._D2
             - self.U_g * self._D)

        # Dirichlet BCs: rows for boundary nodes → zeros
        L[0, :] = 0.0
        L[-1, :] = 0.0

        return L

    def _build_interior_eigensystem(self):
        '''
        Eigendecompose the interior sub-block of L for exponential integration.

        Working with interior nodes only (strip the two Dirichlet boundary
        rows/cols) gives an (N-2)×(N-2) system with no degenerate eigenvalues.
        Since A[0] = A[N-1] = 0 at all times, the boundary columns contribute
        zero to the interior RHS, so L_int = L[1:-1, 1:-1] is exact.

        Returns (vals, vecs, vecs_inv) — eigenvalues (N-2,), eigenvector
        matrix (N-2, N-2), and its inverse.
        '''
        L_int = self._L[1:-1, 1:-1]
        vals, vecs = np.linalg.eig(L_int)
        vecs_inv = np.linalg.inv(vecs)
        return vals, vecs, vecs_inv

    def _solve_exponential(self, t_eval, A_init, forcing):
        '''
        Integrate ∂A/∂t = L A + f using an exponential (ETD1) scheme.

        For each sub-interval [t_n, t_{n+1}] the solution is advanced exactly
        for the linear part and the forcing integral is approximated at the
        interval midpoint (first-order ETD):

            b(t_{n+1}) = exp(Λ Δt) b(t_n)
                         + Δt · φ₁(Λ Δt) · V⁻¹ f(x_int, t_mid)

        where  φ₁(z) = (e^z − 1) / z,  b = V⁻¹ A_int,  and  L_int = V Λ V⁻¹.

        This is unconditionally stable for any Δt and sidesteps the O(N⁴)
        stiffness that makes explicit Runge-Kutta methods impractical for
        Chebyshev discretisations.

        Accuracy in time is O(Δt²) for smooth forcing; refine t_eval for
        impulsive or rapidly-varying forcings (rule of thumb: Δt < σ_t/3).

        @param t_eval:  array, output times (must include t_span[0])
        @param A_init:  complex ndarray shape (N,), initial condition
        @param forcing: Forcing instance or None
        @return: (t_eval, A) where A has shape (N, len(t_eval))
        '''
        vals, vecs, vecs_inv = self._build_interior_eigensystem()

        n_t = len(t_eval)
        A_all = np.zeros((self.N, n_t), dtype=complex)

        # Interior initial condition in modal coordinates
        A_int = A_init[1:-1].copy()
        b = vecs_inv @ A_int
        A_all[1:-1, 0] = A_int

        for i in range(n_t - 1):
            t0, t1 = float(t_eval[i]), float(t_eval[i + 1])
            dt = t1 - t0

            exp_lam = np.exp(vals * dt)     # exp(Λ Δt), shape (N-2,)

            # Exact linear propagation in modal space
            b = exp_lam * b

            # ETD1 forcing correction
            if forcing is not None:
                t_mid = 0.5 * (t0 + t1)
                f_full = np.asarray(forcing(self._x, t_mid), dtype=complex)
                f_int = f_full[1:-1]                    # interior nodes only

                g = vecs_inv @ f_int                    # modal forcing

                # φ₁(z) = (e^z − 1) / z,  with L'Hôpital for |z| → 0
                z = vals * dt
                phi1 = np.where(np.abs(z) < 1e-10,
                                1.0 + z * 0.5,
                                (exp_lam - 1.0) / z)

                b = b + dt * phi1 * g

            A_all[1:-1, i + 1] = vecs @ b

        return t_eval, A_all

    def _rhs(self, t, y, forcing):
        '''
        Right-hand side for scipy.integrate.solve_ivp (fallback path).

        The complex field A ∈ ℂ^N is encoded as  y = [Re(A); Im(A)] ∈ ℝ^{2N}
        because solve_ivp only handles real state vectors.

        Note: for Chebyshev discretisations the operator L is highly stiff
        (condition number ~ N⁴).  Use method='BDF' or method='Radau' with
        this path; never RK45.

        @param t:       float, current time
        @param y:       ndarray shape (2N,), real-encoded state
        @param forcing: Forcing instance or None
        @return:        ndarray shape (2N,), real-encoded dA/dt
        '''
        N = self.N
        A = y[:N] + 1j * y[N:]
        dAdt = self._L @ A

        if forcing is not None:
            dAdt = dAdt + np.asarray(forcing(self._x, t), dtype=complex)

        dAdt[0] = 0.0
        dAdt[-1] = 0.0

        return np.concatenate([dAdt.real, dAdt.imag])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x(self):
        '''Chebyshev collocation grid on [x_min, x_max], shape (N,).'''
        return self._x

    @property
    def t(self):
        '''Output time array from the last solve() call, shape (n_t,).'''
        if self._t is None:
            raise RuntimeError("Call .solve() before accessing t.")
        return self._t

    @property
    def A(self):
        '''
        Complex amplitude field A(x, t), shape (N, n_t).

        Rows index spatial position (Chebyshev ordering: A[0] at x_max,
        A[N-1] at x_min); columns index time.  Use self.x for the grid.
        '''
        if self._A is None:
            raise RuntimeError("Call .solve() before accessing A.")
        return self._A

    @property
    def mu_profile(self):
        '''Growth rate μ(x) evaluated on the collocation grid, shape (N,).'''
        if callable(self.mu):
            return np.asarray(self.mu(self._x), dtype=complex)
        return np.full(self.N, complex(self.mu), dtype=complex)

    @property
    def operator_eigenvalues(self):
        '''
        Eigenvalues of the spatial operator L (interior nodes only).

        Negative real parts indicate stable decay; positive real parts
        indicate local growth.  Returned sorted by real part descending.
        '''
        # Interior block (strip boundary rows/cols which are trivially 0)
        L_int = self._L[1:-1, 1:-1]
        evals = np.linalg.eigvals(L_int)
        return evals[np.argsort(-evals.real)]

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def solve(self, t_span, A0=None, forcing=None, method='exponential',
              t_eval=None, rtol=1e-6, atol=1e-9, **kwargs):
        '''
        Integrate the GL equation from t_span[0] to t_span[1].

        Results are stored in self.t and self.A and accessible as properties.
        Returns self for method chaining:  gl.solve(...).plot_spacetime().

        Parameters
        ----------
        t_span  : (float, float)
            (t_start, t_end) integration interval.
        A0      : None, callable, or array-like
            Initial condition A(x, t_start).
            None     → zero everywhere (default).
            callable → A0(x) evaluated on the Chebyshev grid.
            array    → complex array of length N (Chebyshev ordering).
        forcing : Forcing or None
            External forcing term.  None for autonomous (unforced) evolution.
        method  : str
            Time-integration method.  Default: 'exponential'.

            'exponential'
                ETD1 exponential integrator (recommended).  Diagonalises the
                spatial operator L once (O(N³)) then steps in modal space
                with no stability constraint.  Unconditionally stable for
                any Δt.  Accuracy is O(Δt²) for smooth forcing; use a finer
                t_eval if the forcing varies rapidly (e.g. narrow impulse).

            'BDF' | 'Radau'
                Implicit scipy.integrate.solve_ivp solvers.  Fall back to
                these if the exponential integrator gives poor accuracy for
                a strongly nonlinear or parametrically forced extension.

            'RK45' | 'RK23' | 'DOP853'
                Explicit solvers — DO NOT USE for Chebyshev discretisations.
                The stiffness ratio scales as O(N⁴), making explicit methods
                require Δt ~ N⁻⁴ ≈ 10⁻⁹ for N = 128: they will never finish.

        t_eval  : array-like or None
            Times at which the solution is saved.  If None, 200 evenly spaced
            points spanning t_span are used.
        rtol    : float
            Relative tolerance for scipy solve_ivp methods (default 1e-6).
        atol    : float
            Absolute tolerance for scipy solve_ivp methods (default 1e-9).
        **kwargs
            Additional keyword arguments forwarded to solve_ivp (ignored for
            method='exponential').

        Returns
        -------
        self
        '''
        if forcing is not None and not isinstance(forcing, Forcing):
            raise TypeError("forcing must be a Forcing instance or None.")

        N = self.N
        t0, tf = float(t_span[0]), float(t_span[1])

        if t_eval is None:
            t_eval = np.linspace(t0, tf, 200)
        t_eval = np.asarray(t_eval, dtype=float)

        # --- Initial condition -------------------------------------------
        if A0 is None:
            A_init = np.zeros(N, dtype=complex)
        elif callable(A0):
            A_init = np.asarray(A0(self._x), dtype=complex)
        else:
            A_init = np.asarray(A0, dtype=complex)

        if A_init.shape != (N,):
            raise ValueError(
                f"A0 must have length N = {N}; got shape {A_init.shape}."
            )

        A_init[0] = 0.0
        A_init[-1] = 0.0

        # --- Dispatch to chosen integrator ------------------------------
        if method == 'exponential':
            self._t, self._A = self._solve_exponential(t_eval, A_init, forcing)

        else:
            # scipy solve_ivp path (use BDF or Radau, never RK45)
            y0 = np.concatenate([A_init.real, A_init.imag])
            sol = solve_ivp(
                fun=lambda t, y: self._rhs(t, y, forcing),
                t_span=(t0, tf),
                y0=y0,
                method=method,
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
                **kwargs
            )
            if not sol.success:
                raise RuntimeError(f"solve_ivp failed: {sol.message}")
            self._t = sol.t
            self._A = sol.y[:N] + 1j * sol.y[N:]

        return self

    # ------------------------------------------------------------------
    # Helpers shared across visualisation methods
    # ------------------------------------------------------------------

    def _sort_by_x(self):
        '''Return indices that sort self._x in ascending order.'''
        return np.argsort(self._x)

    def _field_data(self, field):
        '''
        Extract scalar data array from the complex field for plotting.

        @param field: 'abs', 'real', 'imag', or 'phase'
        @return: (data, label) — ndarray shape (N, n_t) and y-axis label
        '''
        if field == 'abs':
            return np.abs(self._A), r'$|A(x,t)|$'
        if field == 'real':
            return self._A.real, r'$\mathrm{Re}(A)$'
        if field == 'imag':
            return self._A.imag, r'$\mathrm{Im}(A)$'
        if field == 'phase':
            return np.angle(self._A), r'$\arg(A)$'
        raise ValueError(
            f"field must be 'abs', 'real', 'imag', or 'phase'; got {field!r}."
        )

    def _require_solved(self):
        if self._A is None:
            raise RuntimeError("Call .solve() first.")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_spacetime(self, field='abs', title=None, figsize=(10, 6),
                       cmap=None, ax=None, vmax=None, show_group_velocity=False):
        '''
        Space-time diagram of the amplitude field  A(x, t).

        Colour encodes the chosen scalar projection of the complex field on
        the (t, x) plane.  Useful for identifying wavepacket propagation,
        convective vs absolute instability, and saturation fronts.

        Parameters
        ----------
        field              : str
            'abs'   — |A(x, t)|                             (default)
            'real'  — Re(A(x, t))
            'imag'  — Im(A(x, t))
            'phase' — arg(A(x, t))
        title              : str, optional
        figsize            : tuple
        cmap               : str, optional  matplotlib colormap.
            Defaults to 'viridis' for 'abs', 'RdBu_r' for 'real'/'imag',
            and 'hsv' for 'phase'.
        ax                 : Axes, optional
        vmax               : float, optional  — upper colour-scale bound
        show_group_velocity: bool
            Overlay a dashed line with slope dx/dt = U_g to visualise the
            group velocity characteristic (default False).
        '''
        self._require_solved()

        data, label = self._field_data(field)
        if cmap is None:
            cmap = {'abs': 'viridis', 'real': 'RdBu_r',
                    'imag': 'RdBu_r', 'phase': 'hsv'}[field]

        sort_idx = self._sort_by_x()
        x_plot = self._x[sort_idx]
        data_plot = data[sort_idx, :]

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)

        vcenter = 0 if field in ('real', 'imag') else None
        vmin = None if field in ('real', 'imag', 'phase') else 0

        im = ax.pcolormesh(
            self._t, x_plot, data_plot,
            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto'
        )

        cb = plt.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(label, fontsize=12)

        if show_group_velocity and self.U_g != 0.0:
            t_mid = 0.5 * (self._t[0] + self._t[-1])
            x_line = self.U_g * (self._t - t_mid)
            ax.plot(self._t, x_line, color='white', linewidth=1.2,
                    linestyle='--', alpha=0.8,
                    label=fr'$U_g = {self.U_g:.3g}$')
            ax.legend(fontsize=10, loc='upper left')

        ax.set_xlabel(r'$t$', fontsize=13)
        ax.set_ylabel(r'$x$', fontsize=13)

        if title is None:
            mu_tag = "μ(x)" if callable(self.mu) else f"μ = {self.mu:.4g}"
            title = (
                f"Ginzburg–Landau — {label}\n"
                f"{mu_tag},  γ = {self.gamma:.4g},  "
                f"$U_g$ = {self.U_g:.3g}"
            )
        ax.set_title(title, fontsize=11)

        if standalone:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_snapshot(self, t_idx=-1, field='abs', title=None,
                      figsize=(8, 4), ax=None):
        '''
        Plot A(x) at a single time index.

        Parameters
        ----------
        t_idx : int
            Column index into self.A (and self.t).  Default: last frame.
        field : str
            'abs', 'real', 'imag', or 'phase'.
        title : str, optional
        figsize : tuple
        ax    : Axes, optional
        '''
        self._require_solved()

        sort_idx = self._sort_by_x()
        x_plot = self._x[sort_idx]
        A_snap = self._A[sort_idx, t_idx]
        data, ylabel = self._field_data(field)
        y_plot = data[sort_idx, t_idx]

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(x_plot, y_plot, color='steelblue', linewidth=2)
        ax.set_xlabel(r'$x$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, alpha=0.25)

        t_val = float(self._t[t_idx])
        if title is None:
            title = (
                f"Ginzburg–Landau snapshot at $t = {t_val:.4g}$"
            )
        ax.set_title(title, fontsize=12)

        if standalone:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_growth_rate(self, title=None, figsize=(8, 4), ax=None):
        '''
        Plot the local growth rate profile μ(x) on the collocation grid.

        Displays both Re(μ) (amplification) and Im(μ) (frequency shift) to
        give a complete picture of the spatially varying instability.

        Parameters
        ----------
        title  : str, optional
        figsize: tuple
        ax     : Axes, optional
        '''
        sort_idx = self._sort_by_x()
        x_plot = self._x[sort_idx]
        mu_plot = self.mu_profile[sort_idx]

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(x_plot, mu_plot.real, color='steelblue', linewidth=2,
                label=r'$\mathrm{Re}(\mu)$ — growth rate')
        ax.plot(x_plot, mu_plot.imag, color='tomato', linewidth=2,
                linestyle='--', label=r'$\mathrm{Im}(\mu)$ — frequency shift')
        ax.axhline(0, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)

        ax.set_xlabel(r'$x$', fontsize=13)
        ax.set_ylabel(r'$\mu(x)$', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)

        if title is None:
            title = "Ginzburg–Landau — Local growth rate μ(x)"
        ax.set_title(title, fontsize=12)

        if standalone:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_operator_spectrum(self, title=None, figsize=(7, 6), ax=None):
        '''
        Plot the eigenvalues of the spatial operator L in the complex plane.

        The rightmost eigenvalue determines the long-time growth or decay
        of the unforced system.  If max Re(λ) > 0 the system is absolutely
        unstable; if max Re(λ) < 0 but the system is convectively unstable
        the response to forcing is wavepacket-like.

        Parameters
        ----------
        title  : str, optional
        figsize: tuple
        ax     : Axes, optional
        '''
        evals = self.operator_eigenvalues

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(evals.real, evals.imag,
                   c=evals.real, cmap='RdYlBu_r',
                   s=14, alpha=0.85, zorder=3)

        unstable = evals[evals.real > 0]
        if len(unstable) > 0:
            ax.scatter(unstable.real, unstable.imag,
                       s=100, marker='*', color='limegreen',
                       edgecolors='black', linewidths=0.5, zorder=6,
                       label=f'Unstable ({len(unstable)})')
            ax.legend(fontsize=10)

        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel(r'$\mathrm{Re}(\lambda)$', fontsize=12)
        ax.set_ylabel(r'$\mathrm{Im}(\lambda)$', fontsize=12)
        ax.grid(True, alpha=0.25)

        if title is None:
            mu_tag = "μ(x)" if callable(self.mu) else f"μ = {self.mu:.4g}"
            title = (
                f"Ginzburg–Landau — Operator spectrum\n"
                f"{mu_tag},  γ = {self.gamma:.4g},  N = {self.N}"
            )
        ax.set_title(title, fontsize=11)

        if standalone:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_frequency_response(self, x_probe=0.0, title=None,
                                 figsize=(8, 4), ax=None):
        '''
        Temporal power spectral density |Â(x_probe, ω)|² at a probe location.

        Computed via FFT of the saved time series A(x_probe, t).
        Useful for identifying dominant response frequencies under harmonic
        or broadband forcing, and for locating resonance peaks.

        Parameters
        ----------
        x_probe : float
            Spatial position to probe (interpolated to nearest grid point).
        title   : str, optional
        figsize : tuple
        ax      : Axes, optional
        '''
        self._require_solved()

        idx = int(np.argmin(np.abs(self._x - x_probe)))
        A_probe = self._A[idx, :]           # complex time series

        dt = float(np.mean(np.diff(self._t)))
        n = len(self._t)

        # A_probe is complex → use fft (not rfft which requires real input).
        # Keep only non-negative frequencies (ω ≥ 0) for the one-sided PSD.
        Ahat = np.fft.fft(A_probe)
        omega = np.fft.fftfreq(n, d=dt) * 2.0 * np.pi
        pos = omega >= 0
        omega = omega[pos]
        psd = np.abs(Ahat[pos]) ** 2 / n

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)

        ax.semilogy(omega, psd, color='steelblue', linewidth=1.5)
        ax.set_xlabel(r'$\omega$', fontsize=13)
        ax.set_ylabel(r'$|\hat{A}(\omega)|^2$', fontsize=12)
        ax.grid(True, alpha=0.25, which='both')

        x_actual = float(self._x[idx])
        if title is None:
            title = (
                f"Ginzburg–Landau — Frequency response at "
                f"$x = {x_actual:.4g}$"
            )
        ax.set_title(title, fontsize=12)

        if standalone:
            plt.tight_layout()
            plt.show()

        return ax

    def plot_overview(self, t_idx=-1, x_probe=0.0, figsize=(14, 10)):
        '''
        Four-panel summary figure:
          top-left   — growth rate μ(x)
          top-right  — operator spectrum in the complex plane
          bottom-left  — space-time diagram |A(x, t)|
          bottom-right — temporal PSD at x_probe

        Parameters
        ----------
        t_idx   : int   — time index for the snapshot (default: last frame)
        x_probe : float — probe location for the PSD panel
        figsize : tuple
        '''
        self._require_solved()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        self.plot_growth_rate(ax=axes[0, 0])
        self.plot_operator_spectrum(ax=axes[0, 1])
        self.plot_spacetime(ax=axes[1, 0])
        self.plot_frequency_response(x_probe=x_probe, ax=axes[1, 1])

        mu_tag = "μ(x)" if callable(self.mu) else f"μ = {self.mu:.4g}"
        fig.suptitle(
            f"Ginzburg–Landau overview — "
            f"{mu_tag},  γ = {self.gamma:.4g},  $U_g$ = {self.U_g:.3g}",
            fontsize=13
        )
        plt.tight_layout()
        plt.show()
        return fig

    def animate(self, field='abs', title=None, interval=50,
                dynamic_scaling=True, figsize=(8, 4)):
        '''
        Animate A(x, t) as a 1-D line plot evolving over time.

        Parameters
        ----------
        field          : str   — 'abs', 'real', 'imag', or 'phase'
        title          : str, optional
        interval       : int   — frame delay in milliseconds (default 50)
        dynamic_scaling: bool  — rescale y-axis each frame (default True)
        figsize        : tuple
        '''
        self._require_solved()

        from matplotlib.animation import FuncAnimation

        sort_idx = self._sort_by_x()
        x_plot = self._x[sort_idx]
        data, ylabel = self._field_data(field)
        data_plot = data[sort_idx, :]

        if title is None:
            title = "Ginzburg–Landau Evolution"

        global_max = float(np.max(np.abs(data_plot))) or 1.0

        fig, ax = plt.subplots(figsize=figsize)
        line, = ax.plot(x_plot, data_plot[:, 0], color='steelblue', linewidth=2)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            fontsize=11, verticalalignment='top')

        ax.set_xlabel(r'$x$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(float(x_plot[0]), float(x_plot[-1]))
        ax.grid(True, alpha=0.25)

        if not dynamic_scaling:
            if field == 'abs':
                ax.set_ylim(0, global_max * 1.1)
            else:
                ax.set_ylim(-global_max * 1.1, global_max * 1.1)

        def _update(frame):
            y = data_plot[:, frame]
            line.set_ydata(y)
            time_text.set_text(f't = {self._t[frame]:.4g}')
            if dynamic_scaling:
                y_max = max(float(np.max(np.abs(y))) * 1.1, 1e-15)
                if field == 'abs':
                    ax.set_ylim(0, y_max)
                else:
                    ax.set_ylim(-y_max, y_max)
            return line, time_text

        anim = FuncAnimation(
            fig, _update, frames=len(self._t), interval=interval, blit=False
        )
        plt.tight_layout()
        plt.show()
        return anim

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        mu_tag = "callable" if callable(self.mu) else f"{self.mu:.4g}"
        return (
            f"GinzburgLandau("
            f"mu={mu_tag}, "
            f"gamma={self.gamma:.4g}, "
            f"U_g={self.U_g:.4g}, "
            f"x_lim={self.x_lim}, "
            f"N={self.N})"
        )
