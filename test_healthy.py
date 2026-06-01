# test_healthy.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from engine_core import engine_derivatives, calculate_oil_pressure
from config import PHYSICS as p

# 1. Define the Duty Cycle (20-minute test drive)
def rpm_profile(t):
    if t < 60: return 0.0          # Engine off
    elif t < 300: return 800.0     # Cold start idle
    elif t < 900: return 2000.0    # Heavy driving
    else: return 800.0             # Cool down idle

def load_profile(t):
    if t < 300: return 0.0         # No load at idle
    elif t < 900: return 0.8       # 80% load while driving
    else: return 0.0

def ode_wrapper(t, y):
    return engine_derivatives(t, y, rpm_profile(t), load_profile(t))

# 2. Setup the Initial State
initial_state = [p["T_ambient"], p["T_ambient"], 0.0] 
t_span = (0, 1200)       # 1200 seconds = 20 minutes
t_eval = np.arange(0, 1200, 0.1) # 10 Hz sampling rate

print("Solving Engine Thermodynamics... (Please wait)")

# 3. Solve the Differential Equations
solution = solve_ivp(
    ode_wrapper, 
    t_span, 
    initial_state, 
    t_eval=t_eval, 
    method='RK45'
)

time = solution.t
T_cool = solution.y[0]
T_oil = solution.y[1]
omega = solution.y[2]
P_oil = calculate_oil_pressure(omega, T_oil, p["C_clear_0"])

# 4. Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(time, omega, label="Engine Speed (RPM)", color='black')
axs[0].set_ylabel("RPM")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(time, T_cool, label="Coolant Temp (C)", color='blue')
axs[1].plot(time, T_oil, label="Oil Temp (C)", color='orange')
axs[1].set_ylabel("Temperature (C)")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(time, P_oil, label="Oil Pressure", color='red')
axs[2].set_ylabel("Pressure (Relative)")
axs[2].set_xlabel("Time (seconds)")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()