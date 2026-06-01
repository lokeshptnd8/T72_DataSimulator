import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from engine_core import engine_derivatives, calculate_oil_pressure, calculate_gearoil_pressure, calculate_electrical
from config import PHYSICS as p

print("--- Phase 1: Ingesting 1-Hour SD Card Data ---")
# 1. Load the real tank telemetry from the SD card
# Assuming the SD card outputs a standard CSV with our 9 channels
sd_data = pd.read_csv("sd_card_1hr_log.csv")

# 2. Extract Real Baselines (Filtering for Warm Idle: RPM < 1000)
idle_data = sd_data[sd_data['engine_rpm'] < 1000]

real_idle_P_oil = idle_data['engine_oil_pressure_kgcm2'].mean()
real_idle_P_gear = idle_data['gearbox_oil_pressure_kgcm2'].mean()
real_voltage_dip = sd_data['battery_voltage_v'].min() # Find lowest crank voltage

# 3. Extract Real Sensor Noise (Standard Deviation)
# We replace our fake Gaussian noise with the actual electrical noise of the tank
noise_profiles = {
    'T_oil': sd_data['engine_oil_temp_c'].std(),
    'P_oil': sd_data['engine_oil_pressure_kgcm2'].std(),
    'Voltage': sd_data['battery_voltage_v'].std()
}

print(f"Real Tank Baseline - Oil Pressure: {real_idle_P_oil:.2f} kg/cm2")
print(f"Real Tank Baseline - Cranking Voltage: {real_voltage_dip:.2f} V")

print("\n--- Phase 2: Reverse-Engineering Physics State ---")
# 4. Calibrate the Initial Wear States based on the SD Card
# If a perfect new engine is 5.8 kg/cm2 (clearance = 1.0)
# We calculate the current wear multiplier of THIS specific tank
current_bearing_wear = np.sqrt(5.8 / real_idle_P_oil) if real_idle_P_oil > 0 else 1.0

# If a perfect battery dips to 38V, we calculate internal sulfation
current_sulfation = 48.0 - real_voltage_dip

print(f"Calibrated Bearing Wear Multiplier: {current_bearing_wear:.4f}")
print(f"Calibrated Battery Sulfation: {current_sulfation:.2f} Ohms")
print("\n--- Phase 3: Generating Future Lifecycle (Rule-Based) ---")

# We start simulation from Hour 1 (since we already have 1 hour of real data)
future_hours = np.arange(1, 200, 10) # Simulate next 200 hours
hybrid_dataset = []

for hour in future_hours:
    # 5. Apply the established degradation rules, but starting from the SD Card baseline
    deg_params = {
        # Bearing wear continues growing from the calibrated baseline
        'bearing_wear': current_bearing_wear * (1 + 0.000003 * (hour ** 2.1)), 
        'h_rad': max(100.0, 1800.0 * (1 - 0.005 * hour)), # Radiator fouling    
        'clutch_wear': max(0.6, 1.0 - 0.0008 * hour),         
        'sulfation': current_sulfation + (0.05 * hour) # Sulfation continues growing
    }
    
    # 6. Run the ODE solver for a simulated diagnostic session
    p["h_rad"] = deg_params['h_rad']
    
    # Simulate a 5-minute (300s) diagnostic run for feature extraction
    sol = solve_ivp(lambda t, y: engine_derivatives(t, y, 800.0, 0.0), 
                    (0, 300), [p["T_ambient"], p["T_ambient"], 0.0, 1200.0], 
                    t_eval=np.arange(0, 300, 1.0), method='RK45')
    
    # 7. Extract the future telemetry
    T_cool, T_oil, omega, V_fuel = sol.y
    RPM = omega
    
    # Apply physics equations
    P_oil = calculate_oil_pressure(omega, T_oil, deg_params['bearing_wear'])
    
    # 8. Inject the REAL noise profile from the SD card
    noisy_P_oil = P_oil + np.random.normal(0, noise_profiles['P_oil'], len(P_oil))
    
    # Log the future forecasted feature (e.g., Warm Idle Oil Pressure)
    hybrid_dataset.append({
        'Future_Engine_Hour': hour,
        'Forecasted_Oil_Pressure': np.mean(noisy_P_oil)
    })

# Export the hybrid forecasted data
hybrid_df = pd.DataFrame(hybrid_dataset)
hybrid_df.to_csv("calibrated_future_forecast.csv", index=False)
print("Simulation Complete. Future trajectory saved to calibrated_future_forecast.csv")