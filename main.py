import numpy as np
from lib import Space, LinNSSol

# Define the Spatial Domain
sp = Space((0, 1), (-0.2, 0.2), 400, 400)

# Define Normal Mode Parameters
# v'(x,y,z,t) = v_hat(y) * exp(i(alpha*x + beta*z - omega*t))
alpha = 10
# Omega = Real + Imaginary*j
# Imaginary > 0 -> Unstable (Growth)
# Imaginary < 0 -> Stable (Damping)
omega = 5 + 2.5*1j

# Time array
t_array = np.linspace(0, 1, 200) 

# Define the Mode Shape function v_hat(y)
def mode_profile(y):
    sigma = 0.05
    return np.exp(-y**(2) / sigma**2) * np.exp(2j * np.pi * y)



# Compute the Solution
ns_sol = LinNSSol(mode_profile, alpha, omega, sp, t_array)

# Visualize
ns_sol.animate_3d(title=f"Unstable Mode (2D): $\\omega={omega.real} + {omega.imag}i$", dynamic_scaling = False)
