import numpy as np

from lib import CouetteFlow, OrrSommerfeld, TransientGrowth, Space

# ============================================================
# Maximum Transient Growth of parallel shear flows
# ============================================================
#
# Problem: find the initial perturbation a (modal coefficients) that
# maximises the kinetic energy at time t, subject to unit initial energy:
#
#   G(t) = max_{a : a^H Q a = 1}  a^H [exp(Λt)]^H Q exp(Λt) a
#
# where:
#   Ψ      = N×M matrix of M selected OS eigenfunctions φ_j(y)
#   Q      = Ψ^H W Ψ   (M×M kinetic-energy Gram matrix,
#                        W = diag of Clenshaw-Curtis weights)
#   Λ      = diag(λ_1, …, λ_M),  λ_j = −iαc_j  (temporal eigenvalues)
#   exp(Λt)= diag(exp(λ_j t))
#
# Solution via SVD:  write Q = F^H F (Cholesky) and set b = Fa:
#
#   G(t) = σ_max( F exp(Λt) F^{-1} )²
#
# The optimal initial condition is  a_opt = F^{-1} v_1  where v_1 is
# the first right singular vector of F exp(Λt) F^{-1}.
# ============================================================

# ------------------------------------------------------------
# Plane Couette Flow — Re=400, α=0.63, β=1.26
# ------------------------------------------------------------

flow = CouetteFlow()
solver = OrrSommerfeld(
    flow=flow,
    Re=400,
    alpha=0.63,
    beta=1.26,
    N=256,
).solve()

print(solver)
print(f"Most unstable eigenvalue: c = {solver.eigenvalues[0]:.6f}")
print(f"Growth rate:              ω_i = {solver.growth_rate:.6f}")

# --- Transient growth -------------------------------------------------------
tg = TransientGrowth(solver, n_modes=50)
print(tg)

t_array = np.linspace(0, 80, 300)
G_max, t_max = tg.peak_growth(t_array)
print(f"\nPeak transient growth: G_max = {G_max:.4f} at t = {t_max:.2f}")

# G(t) curve
tg.plot_growth(t_array)

# Optimal initial perturbation profiles at peak time
tg.plot_optimal_mode(t_max)

# Eigenspectrum + G(t) side by side
tg.plot_spectrum_and_growth(t_array)

# ------------------------------------------------------------
# Animate the transient evolution of the optimal perturbation
# ------------------------------------------------------------
# Spatial domain: one streamwise wavelength (2π/α) × channel height [-1, 1]
lam_x = 2 * np.pi / solver.alpha
sp = Space((0, lam_x), (-1, 1), 300, 150)

# Time array spanning the full growth-then-decay cycle (0 → 2 t_max)
t_anim = np.linspace(0, 2 * t_max, 200)

# 2-D phase/amplitude view (hue = phase, alpha = amplitude)
tg.animate(t_max, sp, t_anim, dynamic_scaling=False)

# 3-D surface view of Re(v)
tg.animate_3d(t_max, sp, t_anim, dynamic_scaling=False)
