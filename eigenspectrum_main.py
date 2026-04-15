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
# Plane Couette Flow — linearly stable for all Re
# ------------------------------------------------------------

F = CouetteFlow()
couette_solver = OrrSommerfeld(
    flow=F,
    Re=400,
    alpha=0.63,
    beta=1.26,
    N=256
).solve()

F.plot()
couette_solver.plot_spectrum()
couette_solver.plot_eigenmode(3)
