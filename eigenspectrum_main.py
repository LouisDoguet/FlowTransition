import numpy as np
from lib import OrrSommerfeld, PoiseuilleFlow, CouetteFlow, CustomFlow

# ============================================================
# Eigenspectrum of parallel shear flows — Orr-Sommerfeld
# Schmid & Henningson (2001), Stability and Transition in Shear Flows
# ============================================================
#
# The Orr-Sommerfeld equation governs the wall-normal velocity
# perturbation φ(y) for a normal mode  v' = φ(y) exp[i(αx + βz - ωt)]:
#
#   [(U - c)(D² - k²) - U''] φ = (1/iαRe)(D² - k²)² φ
#
# Eigenvalue: c = ω/α  (complex phase velocity)
#   Im(c) > 0  →  unstable (growing mode)
#   Im(c) < 0  →  stable   (decaying mode)
#
# Wavenumbers: k² = α² + β²  (β=0 for 2-D perturbations)
# ============================================================


# ------------------------------------------------------------
# 1.  Plane Poiseuille Flow — classical Y-shaped eigenspectrum
# ------------------------------------------------------------
# U(y) = 1 - y²,  walls at y = ±1
# At Re=10000, α=1 one Tollmien-Schlichting mode is unstable.
# Increase N for better-resolved A, P, S branches.

flow = PoiseuilleFlow()

solver = OrrSommerfeld(
    flow=flow,
    Re=10000,     # Reynolds number
    alpha=1.0,    # streamwise wavenumber α
    beta=0.,     # spanwise wavenumber β (0 → pure 2-D perturbation)
    N=128         # Chebyshev resolution — raise to 256 for sharper branches
)

solver.solve()

# Full Y-shaped eigenspectrum in the complex c-plane
solver.plot_spectrum(title=r"Plane Poiseuille — Y-Shaped Eigenspectrum  "
                          r"($Re=10000,\ \alpha=1$)")

# Eigenfunction of the most unstable (Tollmien-Schlichting) mode
solver.plot_eigenmode(mode_index=0)

# Quick status print
print(f"Most unstable c = {solver.eigenvalues[0]:.6f}")
print(f"Temporal growth rate ω_i = α·c_i = {solver.growth_rate:.6f}")
print(f"Flow is {'UNSTABLE' if solver.is_unstable else 'STABLE'}")


# ------------------------------------------------------------
# 2.  Eigenspectrum sweep over Reynolds numbers
#     Watch the Y-shape form and one branch cross into c_i > 0
# ------------------------------------------------------------

solver.plot_spectrum_grid(
    Re_values=[1000, 3000, 5772, 8000, 10000, 15000],
    cols=3,
    figsize=(15, 9)
)


# ------------------------------------------------------------
# 3.  Plane Couette Flow — linearly stable for all Re
# ------------------------------------------------------------

couette_solver = OrrSommerfeld(
    flow=CouetteFlow(),
    Re=1000,
    alpha=1.0,
    N=64
).solve()

couette_solver.plot_spectrum()


# ------------------------------------------------------------
# 4.  Custom flow profile — define U(y) as any callable
# ------------------------------------------------------------

def sinusoidal_profile(y):
    '''Sinusoidal jet-like profile — inflectional (Rayleigh unstable).'''
    return np.sin(np.pi * y / 2)

custom_solver = OrrSommerfeld(
    flow=CustomFlow(sinusoidal_profile, name="Sinusoidal"),
    Re=5000,
    alpha=1.0,
    beta=1.0,
    N=100
).solve()

custom_solver.plot_spectrum()
custom_solver.plot_eigenmode(0)
