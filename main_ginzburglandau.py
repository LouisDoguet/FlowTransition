import numpy as np

from lib.ginzburglandau import (GinzburgLandau,
                                 PointForcing,
                                 HarmonicForcing,
                                 ImpulseForcing,
                                 WavepacketForcing,
                                 CompositeForcing)

# ============================================================
# Ginzburg-Landau equation test cases
# ============================================================
#
# Linearised GL equation:
#
#   ∂A/∂t = μ(x) A + γ ∂²A/∂x² − U_g ∂A/∂x + f(x, t)
#
# where:
#   A(x, t)  complex perturbation amplitude
#   μ(x)     local growth rate  (Re > 0 → locally unstable)
#   γ        complex diffusion / dispersion coefficient
#   U_g      group velocity (downstream advection speed)
#   f(x, t)  external forcing (point source, harmonic, impulse, …)
#
# The four canonical regimes studied below:
#
#  Case 1 — Stable uniform system, harmonic point forcing
#           Steady-state spatial wavepacket; verifies the Green's function.
#
#  Case 2 — Convective instability, harmonic forcing
#           Parabolic μ(x): stable at boundaries, locally unstable at centre.
#           Response: spatially amplified wave travelling downstream.
#
#  Case 3 — Impulse response (Green's function)
#           Single space-time impulse; reveals the causal wavepacket cone.
#           Slope of the cone edges = group / envelope velocities.
#
#  Case 4 — Wavepacket forcing on a convectively unstable background
#           Finite-duration Gaussian wavepacket is launched and amplified.
# ============================================================


# ============================================================
# Case 1 — Stable uniform system + harmonic point forcing
# ============================================================
#
# μ = −0.1  (uniform decay), γ = 1 (real diffusion), U_g = 0.
# HarmonicForcing at x = 0 with ω₀ = 0.8.
#
# At long times A reaches a steady oscillation at ω₀; the spatial shape
# is a decaying exponential on each side of the source — the 1-D Green's
# function of the operator (∂_x² − μ/γ).
#
# Expected output:
#   • Space-time plot: two symmetric decaying tails emanating from x = 0.
#   • Frequency response: sharp peak at ω₀ = 0.8.
# ============================================================

print("=" * 60)
print("Case 1 — Stable system, harmonic forcing")
print("=" * 60)

gl1 = GinzburgLandau(
    mu    = -0.1,
    gamma = 1.0 + 0j,
    U_g   = 0.0,
    x_lim = (-20, 20),
    N     = 128,
)

forcing1 = HarmonicForcing(x0=0.0, omega=0.8, A0=1.0, sigma=0.5)

t_eval1 = np.linspace(0, 60, 300)
gl1.solve(t_span=(0, 60), forcing=forcing1, t_eval=t_eval1)

print(gl1)
print(f"  A max amplitude : {np.max(np.abs(gl1.A)):.4f}")
print(f"  A at x=0, t=60  : {gl1.A[np.argmin(np.abs(gl1.x)), -1]:.4f}")

# Growth rate profile (flat line at −0.1)
gl1.plot_growth_rate(title="Case 1 — Stable uniform growth rate μ = −0.1")

# Operator spectrum (all eigenvalues with Re < 0)
gl1.plot_operator_spectrum(title="Case 1 — Operator spectrum (all stable)")

# Space-time diagram: symmetric tails from source
gl1.plot_spacetime(
    field='abs',
    title="Case 1 — |A(x,t)|  stable system + harmonic forcing (ω₀ = 0.8)",
    show_group_velocity=False,
)

# Snapshot at final time: steady spatial envelope
gl1.plot_snapshot(
    t_idx=-1,
    field='abs',
    title=f"Case 1 — Steady-state |A(x)| at t = {t_eval1[-1]:.0f}",
)

# Frequency response at x = 0: verify peak at ω₀
gl1.plot_frequency_response(
    x_probe=0.0,
    title="Case 1 — PSD at x = 0 (expected peak at ω₀ = 0.8)",
)


# ============================================================
# Case 2 — Convective instability, harmonic forcing
# ============================================================
#
# Parabolic growth rate:
#   μ(x) = μ₀ + μ₂ (1 − (x/L)²)
#
# With μ₀ = −0.2, μ₂ = 0.25, L = 10:
#   μ(0) = +0.05  (unstable at centre)
#   μ(±10) = −0.2 (stable at boundaries → Dirichlet BCs consistent)
#
# Group velocity U_g = 2 advects the response downstream.
# HarmonicForcing at x = −5 (upstream of the unstable region).
#
# Expected output:
#   • Space-time plot: oblique stripes with slope ~ U_g in the
#     unstable region, decaying downstream.
#   • PSD at x = +5: amplified harmonic peak at ω₀.
# ============================================================

print("\n" + "=" * 60)
print("Case 2 — Convective instability, harmonic forcing")
print("=" * 60)

mu0, mu2, L_mu = -0.2, 0.25, 10.0
mu_conv = lambda x: mu0 + mu2 * (1.0 - (x / L_mu) ** 2)

gl2 = GinzburgLandau(
    mu    = mu_conv,
    gamma = 1.0 + 0.2j,
    U_g   = 2.0,
    x_lim = (-20, 20),
    N     = 128,
)

forcing2 = HarmonicForcing(x0=-5.0, omega=0.5, A0=1.0, sigma=0.8)

t_eval2 = np.linspace(0, 50, 300)
gl2.solve(t_span=(0, 50), forcing=forcing2, t_eval=t_eval2)

print(gl2)
print(f"  μ(0) = {mu_conv(0.0):.3f}  (should be +{mu2+mu0:.3f}, unstable)")
print(f"  μ(±10) = {mu_conv(10.0):.3f}  (stable at boundaries)")
print(f"  A max amplitude: {np.max(np.abs(gl2.A)):.4f}")

# Growth rate profile: shows the unstable pocket
gl2.plot_growth_rate(title="Case 2 — Convective instability: parabolic μ(x)")

# Space-time diagram: amplified oblique stripes
gl2.plot_spacetime(
    field='abs',
    title="Case 2 — Convective instability |A(x,t)|  (U_g = 2, ω₀ = 0.5)",
    show_group_velocity=True,
)

# Real part shows wave crests propagating downstream
gl2.plot_spacetime(
    field='real',
    title="Case 2 — Re(A(x,t))  wave crests",
    show_group_velocity=True,
)

# Snapshot near peak time
t_peak2_idx = int(np.argmax(np.max(np.abs(gl2.A), axis=0)))
gl2.plot_snapshot(
    t_idx=t_peak2_idx,
    field='abs',
    title=f"Case 2 — |A(x)| at peak time t = {gl2.t[t_peak2_idx]:.2f}",
)

# PSD downstream of the unstable zone
gl2.plot_frequency_response(
    x_probe=5.0,
    title="Case 2 — PSD at x = +5  (amplified peak at ω₀ = 0.5)",
)

# Operator spectrum: some eigenvalues should have Re > 0
gl2.plot_operator_spectrum(
    title="Case 2 — Operator spectrum (convectively unstable pocket)",
)


# ============================================================
# Case 3 — Impulse response (causal Green's function)
# ============================================================
#
# μ = −0.05  (very mildly stable), γ = 1, U_g = 1.5.
# ImpulseForcing at x = 0, t₀ = 1.
#
# The response is the fundamental solution G(x, t; 0, t₀) of the GL
# operator.  For a constant-coefficient system with group velocity U_g,
# the wavepacket centre travels at x_c = U_g · (t − t₀) and the
# envelope spreads diffusively (σ ~ √(2|γ|(t−t₀))).
#
# Expected output:
#   • Space-time plot: a cone originating from (0, t₀ = 1), with
#     the apex tracking  x = U_g · t.
#   • Snapshot at late time: Gaussian envelope centred on x_c.
# ============================================================

print("\n" + "=" * 60)
print("Case 3 — Impulse response / Green's function")
print("=" * 60)

gl3 = GinzburgLandau(
    mu    = -0.05,
    gamma = 1.0 + 0j,
    U_g   = 1.5,
    x_lim = (-10, 25),
    N     = 128,
)

t0_impulse = 1.0
forcing3 = ImpulseForcing(x0=0.0, t0=t0_impulse, A0=1.0,
                           sigma_x=0.4, sigma_t=0.1)

t_eval3 = np.linspace(0, 12, 300)
gl3.solve(t_span=(0, 12), forcing=forcing3, t_eval=t_eval3)

print(gl3)
print(f"  A max amplitude: {np.max(np.abs(gl3.A)):.4f}")

# Theoretical centre position at t = 10: x_c = U_g * (t - t0)
t_check = 10.0
x_centre_theory = gl3.U_g * (t_check - t0_impulse)
print(f"  Expected wavepacket centre at t={t_check:.0f}: "
      f"x_c = {x_centre_theory:.2f}")

# Space-time diagram: the wavepacket cone
gl3.plot_spacetime(
    field='abs',
    title=("Case 3 — Impulse response |A(x,t)|  "
           f"(U_g = {gl3.U_g}, t₀ = {t0_impulse})"),
    show_group_velocity=True,
)

# Snapshot at t = 10: Gaussian envelope
t_idx3 = int(np.argmin(np.abs(gl3.t - t_check)))
gl3.plot_snapshot(
    t_idx=t_idx3,
    field='abs',
    title=(f"Case 3 — Impulse response |A(x)| at t = {gl3.t[t_idx3]:.2f}  "
           f"(theory centre: x = {x_centre_theory:.2f})"),
)

# Phase of the wavepacket at the same time
gl3.plot_snapshot(
    t_idx=t_idx3,
    field='real',
    title=f"Case 3 — Re(A(x)) at t = {gl3.t[t_idx3]:.2f}",
)


# ============================================================
# Case 4 — Wavepacket forcing on a convectively unstable background
# ============================================================
#
# Parabolic μ(x) with absolute instability threshold just below zero
# so the system is convectively (not absolutely) unstable.
#
# A finite-duration Gaussian wavepacket with carrier k₀ = 0.8 is
# launched from x = −8 at t = 0 and switches off at t = T_on.
# The response wavepacket propagates into the unstable region,
# is amplified, then decays as it reaches the stable boundary region.
#
# Expected output:
#   • Space-time plot: wavepacket enters from the left, amplifies near
#     the unstable centre, then exits to the right.
#   • Composite forcing: combine WavepacketForcing with a steady
#     HarmonicForcing to show superposition.
# ============================================================

print("\n" + "=" * 60)
print("Case 4 — Wavepacket forcing on convectively unstable background")
print("=" * 60)

mu_wp = lambda x: -0.15 + 0.18 * (1.0 - (x / 12.0) ** 2)

gl4 = GinzburgLandau(
    mu    = mu_wp,
    gamma = 1.0 + 0.15j,
    U_g   = 1.8,
    x_lim = (-20, 20),
    N     = 128,
)

# Wavepacket: on for t ∈ [0, T_on], then zero
T_on = 8.0
gate = lambda t: 1.0 if t <= T_on else 0.0

forcing4_wp = WavepacketForcing(
    x0=- 8.0, k0=0.8, sigma=2.0, g=gate, A0=1.0
)

# Combine with a weak steady harmonic source at x = 0 to show superposition
forcing4_harm = HarmonicForcing(x0=0.0, omega=0.4, A0=0.1, sigma=0.6)

forcing4 = CompositeForcing(forcing4_wp, forcing4_harm)
print(forcing4)

t_eval4 = np.linspace(0, 40, 300)
gl4.solve(t_span=(0, 40), forcing=forcing4, t_eval=t_eval4)

print(gl4)
print(f"  A max amplitude : {np.max(np.abs(gl4.A)):.4f}")
print(f"  μ(0) = {mu_wp(0.0):.3f}  (unstable centre)")
print(f"  μ(±12) = {mu_wp(12.0):.3f}  (stable boundaries)")

# Growth rate profile
gl4.plot_growth_rate(
    title="Case 4 — Parabolic growth rate μ(x) (convective instability)",
)

# Space-time diagram: full wavepacket evolution
gl4.plot_spacetime(
    field='abs',
    title=("Case 4 — Wavepacket |A(x,t)|  "
           f"(T_on = {T_on}, k₀ = 0.8, U_g = {gl4.U_g})"),
    show_group_velocity=True,
)

# Real part: wave crests inside the wavepacket
gl4.plot_spacetime(
    field='real',
    title="Case 4 — Re(A(x,t))  wave crests inside wavepacket",
)

# PSD at x = +5: shows both the harmonic source and broadband wavepacket
gl4.plot_frequency_response(
    x_probe=5.0,
    title="Case 4 — PSD at x = +5  (harmonic peak + wavepacket broadband)",
)

# Four-panel overview summary for Case 4
gl4.plot_overview(x_probe=5.0)
