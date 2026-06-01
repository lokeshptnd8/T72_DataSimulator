# final_generator.py
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from engine_core import engine_derivatives, calculate_oil_pressure, calculate_gearoil_pressure, calculate_electrical
from config import PHYSICS as p
import argparse # NEW: For professional command-line arguments

# --- CLI ARGUMENT PARSER ---
parser = argparse.ArgumentParser(description="T-72 Synthetic Telemetry Generator")
parser.add_argument('--temp', type=float, default=25.0, help="Ambient external temperature in Celsius")
parser.add_argument('--humidity', type=float, default=40.0, help="External relative humidity percentage")
args = parser.parse_args()

print(f"\n--- Initializing Simulator Environment ---")
print(f"Ambient Temperature set to: {args.temp}°C")
print(f"Relative Humidity set to: {args.humidity}%")
print(f"------------------------------------------\n")

# Apply the custom inputs to the Physics parameters
from config import PHYSICS as p
p["T_ambient"] = args.temp

# Calculate Air Density Modifier based on Humidity
# High humidity slightly lowers air density, affecting radiator cooling efficiency
humidity_penalty = 1.0 - (args.humidity * 0.0005) 
p["h_rad_base"] = p.get("h_rad", 1800.0) * humidity_penalty

# 1. The Precision Duty Cycle
def rpm_profile(t):
    if t < 5: return 0.0          # Off
    elif t < 10: return 200.0     # Cranking
    elif t < 300: return 800.0    # Warm Idle (for Lubrication)
    elif t < 900: return 2000.0   # Heavy Load (for Thermal Soak)
    elif t < 1500: return 1750.0  # Steady Cruise (for Gearbox)
    else: return 0.0

def load_profile(t):
    if 300 <= t < 900: return 0.9 
    elif 900 <= t < 1500: return 0.6
    else: return 0.0

def extract_ml_features(time, states, degradation_params):
    T_cool, T_oil, omega, V_fuel = states
    RPM = omega # REMOVED 9.549 multiplier
    
    P_oil = calculate_oil_pressure(omega, T_oil, degradation_params['bearing_wear'])
    P_gear = calculate_gearoil_pressure(omega, degradation_params['clutch_wear'])
    _, Voltage, _ = calculate_electrical(omega, degradation_params['sulfation'])

    features = {}

    # 1. Lubrication Degradation (XGBoost)
    valid_lube_idx = np.where((RPM < 1000) & (T_oil < 80))[0]
    if len(valid_lube_idx) > 0:
        features['Warm_Idle_Oil_Pressure'] = np.mean(P_oil[valid_lube_idx])
    else:
        features['Warm_Idle_Oil_Pressure'] = 0.0

    # 2. Thermal Soak Rate (LSTM)
    valid_thermal_idx = np.where(RPM > 1600)[0]
    if len(valid_thermal_idx) > 0:
        heavy_coolant = T_cool[valid_thermal_idx]
        heavy_time = time[valid_thermal_idx]
        
        cross_70 = np.where(heavy_coolant >= 70)[0]
        cross_100 = np.where(heavy_coolant >= 100)[0]
        
        if len(cross_70) > 0 and len(cross_100) > 0:
            features['Thermal_Soak_Time_Minutes'] = (heavy_time[cross_100[0]] - heavy_time[cross_70[0]]) / 60.0
        else:
            features['Thermal_Soak_Time_Minutes'] = 25.0 
    else:
        features['Thermal_Soak_Time_Minutes'] = 25.0

    # 3. Transmission Clutch Pack Fatigue (RSF)
    valid_gear_idx = np.where((RPM >= 1600) & (RPM <= 1900) & (time >= 1200))[0]
    if len(valid_gear_idx) > 0:
        features['Cruise_Gearbox_Pressure'] = np.mean(P_gear[valid_gear_idx])
    else:
        features['Cruise_Gearbox_Pressure'] = 0.0

    # 4. Battery Bank Sulfation (GPR)
    crank_idx = np.where((time >= 5) & (time <= 10))[0]
    if len(crank_idx) > 0:
        features['Cranking_Voltage_Dip'] = np.min(Voltage[crank_idx])
    else:
        features['Cranking_Voltage_Dip'] = 48.0

    return features


# --- MASTER GENERATOR ---
N_ENGINES = 4
engine_hours = np.arange(0, 501, 10) 

dataset_ml = []
dataset_raw_telemetry = [] # NEW: Array to hold the second-by-second 9-channel data

print("Running Final Generator: Extracting ML Features AND Raw 9-Channel Telemetry...")

for unit in range(N_ENGINES):
    print(f"Processing Virtual Engine {unit}...")
    
    for hour in engine_hours:
        # 1. The Degradation Physics
        deg_params = {
            'bearing_wear': 1.0 * (1 + 0.000003 * (hour ** 2.1)), 
            # UPDATED: Incorporate the humidity-modified base radiator cooling
            'h_rad': max(100.0, p["h_rad_base"] * (1 - 0.005 * hour)),     
            'clutch_wear': max(0.6, 1.0 - 0.0008 * hour),         
            'sulfation': 10.0 + (0.05 * hour)                     
        }
        p["h_rad"] = deg_params['h_rad']
        
        # 2. Run the ODE Session (25 minutes at 1 Hz)
        t_eval = np.arange(0, 1500, 1.0)
        sol = solve_ivp(lambda t, y: engine_derivatives(t, y, rpm_profile(t), load_profile(t)), 
                        (0, 1500), [p["T_ambient"], p["T_ambient"], 0.0, 1200.0], 
                        t_eval=t_eval, method='RK45')
        
        # 3. Calculate ALL 9 Channels for the entire 1500 seconds
        T_cool, T_oil, omega, V_fuel = sol.y
        RPM = omega
        P_oil = calculate_oil_pressure(omega, T_oil, deg_params['bearing_wear'])
        P_gear = calculate_gearoil_pressure(omega, deg_params['clutch_wear'])
        
        # We calculate electrical, but discard Alt_Freq (_) as it's not in the new spec
        _, Voltage, Current = calculate_electrical(omega, deg_params['sulfation'])
        
        # Convert Fuel Liters to Percentage (0% to 100%)
        Fuel_Percent = (V_fuel / p["fuel_capacity_0"]) * 100.0
        
        # Engine Hour Meter (Cumulative base hour + the fraction of the current session in hours)
        # sol.t is the time array in seconds.
        Engine_Hours = hour + (sol.t / 3600.0)
        
        # ---> UPDATED: Save the raw 9-channel telemetry matching the EXACT hardware pinout <---
        # Shape: [1500 time steps, 9 sensor channels]
        session_telemetry = np.column_stack((
            T_oil,          # CH1: engine_oil_temp_c
            T_cool,         # CH2: coolant_temp_c
            P_oil,          # CH3: engine_oil_pressure_kgcm2
            P_gear,         # CH4: gearbox_oil_pressure_kgcm2
            RPM,            # CH5: engine_rpm
            Current,        # CH6: battery_current_a
            Voltage,        # CH7: battery_voltage_v
            Fuel_Percent,   # CH8: fuel_level_percent
            Engine_Hours    # CH9: engine_hours
        ))
        dataset_raw_telemetry.append(session_telemetry)

        # 4. Extract the exact ML targets
        ml_features = extract_ml_features(sol.t, sol.y, deg_params)
        
        # Inject Anomalies 
        if hour == 200 and unit == 0: ml_features['Warm_Idle_Oil_Pressure'] = 2.8 
        if hour == 400 and unit == 1: ml_features['Cranking_Voltage_Dip'] = 31.0
            
        dataset_ml.append({
            'Unit': unit,
            'Engine_Hour': hour,
            **ml_features
        })

# Export 1: The ML Features (CSV)
df = pd.DataFrame(dataset_ml)
df.to_csv("t72_ml_features.csv", index=False)

# Export 2: The Raw 9-Channel Telemetry (NPZ)
# Reshape into a 3D array: [Total Sessions, 1500 seconds, 9 channels]
raw_matrix = np.array(dataset_raw_telemetry)
np.savez("t72_raw_9channel_telemetry.npz", telemetry=raw_matrix)

print("\n--- Export Complete ---")
print("1. t72_ml_features.csv saved (For XGBoost/LSTM training)")
print(f"2. t72_raw_9channel_telemetry.npz saved. Matrix shape: {raw_matrix.shape} (For Database/UI)")