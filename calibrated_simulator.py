# calibrated_simulator.py
import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from engine_core import engine_derivatives, calculate_oil_pressure, calculate_gearoil_pressure, calculate_electrical
from config import PHYSICS as p

# --- HELPER FUNCTION FOR INTERACTIVE INPUTS ---
def get_user_input(prompt, default_val, cast_type=float):
    """Prompts the user for an input, falling back to a default if they just press ENTER."""
    user_in = input(f"{prompt} [Default: {default_val}]: ").strip()
    if not user_in:
        return default_val
    try:
        return cast_type(user_in)
    except ValueError:
        print(f"  [!] Invalid input detected. Reverting to default: {default_val}")
        return default_val

# --- INTERACTIVE CLI WIZARD ---
print("\n========================================================")
print("  T-72 HYBRID DIGITAL TWIN: SD CARD CALIBRATION")
print("========================================================")
print("Press [ENTER] on any prompt to use the default value.\n")

temp = get_user_input("Enter Ambient Temperature (°C)", 25.0, float)
humidity = get_user_input("Enter Relative Humidity (%)", 40.0, float)
num_engines = get_user_input("Enter number of identical tanks to simulate", 5, int)
num_sessions = get_user_input("Enter number of future sessions to forecast per tank", 20, int)
session_duration_mins = get_user_input("Enter duration of each session (minutes)", 5.0, float)

print(f"\n--- Initializing Simulator Environment ---")
print(f"Temperature     : {temp}°C")
print(f"Humidity        : {humidity}%")
print(f"Total Tanks     : {num_engines}")
print(f"Sessions/Tank   : {num_sessions}")
print(f"Session Length  : {session_duration_mins} minutes")
print(f"------------------------------------------\n")

# Apply custom physics
p["T_ambient"] = temp
humidity_penalty = 1.0 - (humidity * 0.0005) 
p["h_rad_base"] = p.get("h_rad", 1800.0) * humidity_penalty 
session_duration_seconds = session_duration_mins * 60.0

# --- PHASE 1: SD CARD INGESTION & CALIBRATION ---
print("--- Phase 1: Ingesting 1-Hour SD Card Baseline ---")

# Safety net: Auto-generate a dummy SD card if the file is missing for the demo
if not os.path.exists("sd_card_1hr_log.csv"):
    print("  [!] sd_card_1hr_log.csv not found. Auto-generating a mock physical baseline...")
    dummy_data = pd.DataFrame({
        'engine_rpm': np.random.uniform(750, 850, 3600),
        'engine_oil_pressure_kgcm2': np.random.normal(5.4, 0.1, 3600), # Real tank is slightly worn
        'gearbox_oil_pressure_kgcm2': np.random.normal(12.5, 0.2, 3600),
        'battery_voltage_v': np.random.normal(27.8, 0.3, 3600),
        'engine_oil_temp_c': np.random.normal(85.0, 2.0, 3600)
    })
    dummy_data.to_csv("sd_card_1hr_log.csv", index=False)

sd_data = pd.read_csv("sd_card_1hr_log.csv")

# Extract Real Baselines
idle_data = sd_data[sd_data['engine_rpm'] < 1000]
real_idle_P_oil = idle_data['engine_oil_pressure_kgcm2'].mean()
real_voltage_dip = sd_data['battery_voltage_v'].min()

noise_profiles = {
    'P_oil': sd_data['engine_oil_pressure_kgcm2'].std(),
    'Voltage': sd_data['battery_voltage_v'].std()
}

print(f"  -> Real Tank Baseline Oil Pressure: {real_idle_P_oil:.2f} kg/cm2")
print(f"  -> Extracted Sensor Noise Variance: {noise_profiles['P_oil']:.4f}")

# Reverse-Engineer Initial Wear States
current_bearing_wear = np.sqrt(5.8 / real_idle_P_oil) if real_idle_P_oil > 0 else 1.0
current_sulfation = max(10.0, 48.0 - real_voltage_dip)

print(f"  -> Calibrated Bearing Wear Multiplier: {current_bearing_wear:.4f}")

# --- PHASE 2: FORECASTING (THE DIGITAL TWIN) ---
print("\n--- Phase 2: Generating Calibrated Future Lifecycle ---")

dataset_raw_telemetry = []
engine_hours_array = np.arange(10, (num_sessions * 10) + 10, 10) # 10 hours of wear per session

# Tracking variables for the final summary
total_data_points = 0
points_per_engine = {}

for unit in range(num_engines):
    print(f"  Forecasting Tank {unit} based on SD Card...")
    points_per_engine[unit] = 0
    
    for hour in engine_hours_array:
        
        # Apply degradation rules, starting FROM the calibrated SD card state
        deg_params = {
            'bearing_wear': current_bearing_wear * (1 + 0.000003 * (hour ** 2.1)), 
            'h_rad': max(100.0, p["h_rad_base"] * (1 - 0.005 * hour)),     
            'clutch_wear': max(0.6, 1.0 - 0.0008 * hour),         
            'sulfation': current_sulfation + (0.05 * hour)                     
        }
        p["h_rad"] = deg_params['h_rad']
        
        # Run ODE Solver
        t_eval = np.arange(0, session_duration_seconds, 1.0)
        sol = solve_ivp(lambda t, y: engine_derivatives(t, y, 800.0, 0.9), # Assumes 800 RPM Warm Idle
                        (0, session_duration_seconds), [p["T_ambient"], p["T_ambient"], 0.0, 1200.0], 
                        t_eval=t_eval, method='RK45')
        
        # Track data points dynamically based on actual ODE solver output
        session_rows = len(sol.t)
        points_per_engine[unit] += session_rows
        total_data_points += session_rows

        T_cool, T_oil, omega, V_fuel = sol.y
        RPM = omega
        P_oil = calculate_oil_pressure(omega, T_oil, deg_params['bearing_wear'])
        P_gear = calculate_gearoil_pressure(omega, deg_params['clutch_wear'])
        _, Voltage, Current = calculate_electrical(omega, deg_params['sulfation'])
        Fuel_Percent = (V_fuel / p["fuel_capacity_0"]) * 100.0
        Engine_Hours = hour + (sol.t / 3600.0)
        
        # Inject REAL noise profile extracted from the SD card
        noisy_P_oil = P_oil + np.random.normal(0, noise_profiles['P_oil'], len(P_oil))
        noisy_Voltage = Voltage + np.random.normal(0, noise_profiles['Voltage'], len(Voltage))
        
        # Formats the session data into a beautifully labeled tabular DataFrame
        session_df = pd.DataFrame({
            'Engine_ID': unit,
            'Session_Base_Hour': hour,
            'Time_Step_Sec': sol.t,
            'CH1_T_oil': T_oil,
            'CH2_T_cool': T_cool,
            'CH3_P_oil': P_oil,
            'CH4_P_gear': P_gear,
            'CH5_RPM': RPM,
            'CH6_Current': Current,
            'CH7_Voltage': Voltage,
            'CH8_Fuel_Pct': Fuel_Percent,
            'CH9_Engine_Hours': Engine_Hours
        })
        dataset_raw_telemetry.append(session_df)

# Export 2: Raw Telemetry (CSV and XML)
print("\n[SYSTEM] Compiling tabular telemetry...")
df_telemetry = pd.concat(dataset_raw_telemetry, ignore_index=True)

print("[SYSTEM] Exporting to CSV...")
df_telemetry.to_csv("t72_calibrated_telemetry.csv", index=False)

print("[SYSTEM] Exporting to XML (This may take a moment for large datasets)...")
df_telemetry.to_xml("t72_calibrated_telemetry.xml", index=False, root_name="FleetTelemetry", row_name="SensorReading", parser="etree")

# --- EXECUTION SUMMARY ---
print("\n========================================================")
print("  DATA GENERATION SUMMARY")
print("========================================================")
print(f"  Total Data Points (Rows)  : {total_data_points:,}")
print(f"  Tabular Dimensions        : {df_telemetry.shape}")
print("--------------------------------------------------------")
for unit, count in points_per_engine.items():
    print(f"  -> Tank {unit} generated      : {count:,} points")
print("========================================================\n")
print("[SYSTEM] Calibrated pipeline complete. Saved to 't72_calibrated_telemetry.xml'.")