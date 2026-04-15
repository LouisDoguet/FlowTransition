import numpy as np

from lib import CouetteFlow, PoiseuilleFlow, OrrSommerfeld, TransientGrowth

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
#   Λ      = diag(σ_1, …, σ_M),  σ_j = i α c_j  (temporal eigenvalues)
#   exp(Λt)= diag(exp(σ_j t))
#
# Solution via SVD:  write Q = F^H F (Cholesky) and set b = Fa:
#
#   G(t) = σ_max( F exp(Λt) F^{-1} )²
#
# The optimal initial condition is  a_opt = F^{-1} v_1  where v_1 is
# the first right singular vector of F exp(Λt) F^{-1}.
# ============================================================


# ------------------------------------------------------------
# Plane Couette Flow — linearly stable, exhibits strong transient growth
# ------------------------------------------------------------

couette = CouetteFlow()
solver = OrrSommerfeld(
    flow=couette,
    Re=1000,
    alpha=1.0,
    beta=0.0,
    N=128,
).solve()

print(solver)
print(f"Most unstable eigenvalue: c = {solver.eigenvalues[0]:.6f}")
print(f"Growth rate:              σ_i = {solver.growth_rate:.6f}")

# --- Transient growth with a selected number of modes --------------------
tg = TransientGrowth(solver, n_modes=50)
print(tg)

t_array = np.linspace(0, 80, 300)
G_max, t_max = tg.peak_growth(t_array)
print(f"\nPeak transient growth: G_max = {G_max:.4f} at t = {t_max:.2f}")

# Plot G(t)
tg.plot_growth(t_array)

# Plot the optimal initial perturbation at peak time
tg.plot_optimal_mode(t_max)

# Side-by-side: eigenspectrum + growth curve
tg.plot_spectrum_and_growth(t_array)


# ------------------------------------------------------------
# Plane Poiseuille Flow — effect of n_modes on the growth curve
# ------------------------------------------------------------

poiseuille = PoiseuilleFlow()
solver_p = OrrSommerfeld(
    flow=poiseuille,
    Re=5000,
    alpha=1.0,
    beta=0.0,
    N=128,
).solve()

t_array_p = np.linspace(0, 50, 300)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))

for n_modes in [10, 30, 60, 128]:
    tg_p = TransientGrowth(solver_p, n_modes=n_modes)
    G_arr = tg_p.compute(t_array_p)
    ax.plot(t_array_p, G_arr, label=f"M = {n_modes}")

ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$G(t)$', fontsize=13)
ax.set_title(
    f"{poiseuille.name} — Transient Growth vs number of modes\n"
    f"Re = {solver_p.Re},  α = {solver_p.alpha}"
)
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()
