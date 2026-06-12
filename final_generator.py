# final_generator.py
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
print("  T-72 DIGITAL TWIN: INTERACTIVE SIMULATION SETUP")
print("========================================================")
print("Press [ENTER] on any prompt to use the default value.\n")

temp = get_user_input("Enter Ambient Temperature (°C)", 25.0, float)
humidity = get_user_input("Enter Relative Humidity (%)", 40.0, float)
num_engines = get_user_input("Enter number of engines to simulate", 5, int)
num_sessions = get_user_input("Enter number of sessions per engine", 20, int)
session_duration_mins = get_user_input("Enter duration of each session (minutes)", 5.0, float)

print(f"\n--- Initializing Simulator Environment ---")
print(f"Temperature     : {temp}°C")
print(f"Humidity        : {humidity}%")
print(f"Total Engines   : {num_engines}")
print(f"Sessions/Engine : {num_sessions}")
print(f"Session Length  : {session_duration_mins} minutes")
print(f"------------------------------------------\n")

# --- APPLY INPUTS TO PHYSICS ENGINE ---
p["T_ambient"] = temp
humidity_penalty = 1.0 - (humidity * 0.0005) 
p["h_rad_base"] = p.get("h_rad", 1800.0) * humidity_penalty 

# Calculate the actual simulation time limits based on user input
session_duration_seconds = session_duration_mins * 60.0

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
    RPM = omega 
    
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
engine_hours = np.arange(0, num_sessions * 10, 10) 

dataset_ml = []
dataset_raw_telemetry = [] 

print("Running Final Generator: Extracting ML Features AND Raw 9-Channel Telemetry...")

# --- DATA TRACKING VARIABLES ---
total_data_points = 0
points_per_engine = {}

for unit in range(num_engines):
    print(f"Processing Virtual Engine {unit}...")
    points_per_engine[unit] = 0
    
    # 1. Randomize the onset of degradation (The Healthy Plateau)
    # The engine will remain perfectly healthy until this specific hour
    onset_hour = np.random.randint(50, 100) 
    
    for hour in engine_hours:
        
        # 2. Calculate active wear: 0 if we haven't reached the onset hour yet
        active_wear_h = max(0, hour - onset_hour)
        
        # 3. The Degradation Physics (Now using active_wear_h instead of total hour)
        deg_params = {
            'bearing_wear': 1.0 * (1 + 0.000003 * (active_wear_h ** 2.1)), 
            'h_rad': max(100.0, p.get("h_rad_base", 1800.0) * (1 - 0.005 * active_wear_h)),     
            'clutch_wear': max(0.6, 1.0 - 0.0008 * active_wear_h),         
            'sulfation': 10.0 + (0.05 * active_wear_h)                     
        }
        p["h_rad"] = deg_params['h_rad']
        
        # ... (ODE solver and the rest of the loop remains exactly the same) ...
        
        # 2. Run the ODE Session dynamically based on user input duration
        t_eval = np.arange(0, session_duration_seconds, 1.0)
        sol = solve_ivp(lambda t, y: engine_derivatives(t, y, rpm_profile(t), load_profile(t)), 
                        (0, session_duration_seconds), [p["T_ambient"], p["T_ambient"], 0.0, 1200.0], 
                        t_eval=t_eval, method='RK45')
        
        # --- TRACK POINTS DYNAMICALLY ---
        session_rows = len(sol.t)
        points_per_engine[unit] += session_rows
        total_data_points += session_rows
        
        # 3. Calculate ALL 9 Channels
        T_cool, T_oil, omega, V_fuel = sol.y
        RPM = omega
        P_oil = calculate_oil_pressure(omega, T_oil, deg_params['bearing_wear'])
        P_gear = calculate_gearoil_pressure(omega, deg_params['clutch_wear'])
        _, Voltage, Current = calculate_electrical(omega, deg_params['sulfation'])
        Fuel_Percent = (V_fuel / p["fuel_capacity_0"]) * 100.0
        Engine_Hours = hour + (sol.t / 3600.0)

        # ====================================================================
        # --- ACUTE ANOMALY INJECTION BLOCK ---
        # Explicitly breaking the healthy operational bounds to train the ML model.
        # This gives your models actual failure data to learn from.
        # ====================================================================

        # Anomaly 1: Severe Overheating (Violating 70-100°C bound)
        if unit == 0 and hour == 300:
            # Simulate a stuck thermostat in the last 5 minutes (300 seconds)
            T_cool[-300:] = np.linspace(T_cool[-300], 118.0, 300) 
            T_oil[-300:] = np.linspace(T_oil[-300], 125.0, 300)   

        # Anomaly 2: Catastrophic Lubrication Failure (Violating 5-10 kg/cm² bound)
        if unit == 1 and hour == 400:
            # P_oil drops entirely out of the healthy range to ~2.5 kg/cm²
            P_oil = P_oil * 0.4 

        # Anomaly 3: Alternator Failure (Violating 22-28V operational bound)
        if unit == 2 and hour == 250:
            # Alternator dies mid-session. Voltage sags to battery baseline.
            is_running = RPM >= 400
            Voltage[is_running] = np.linspace(24.0, 19.5, len(Voltage[is_running])) 

        # Anomaly 4: Transmission RSF Pump Failure (Violating 10-11.5 kg/cm² cruise bound)
        if unit == 3 and hour == 350:
            # Gearbox pressure fails to build during cruise RPMs
            cruise_idx = np.where((RPM >= 1600) & (RPM <= 1900))[0]
            P_gear[cruise_idx] = np.random.normal(7.5, 0.5, len(cruise_idx)) 

        # Anomaly 5: Starter Motor Short Circuit (Violating 42-48V cranking bound)
        if unit == 4 and hour == 150:
            # Starter pulls too much amperage, dragging the series voltage down severely
            crank_idx = np.where((RPM > 0) & (RPM < 400))[0]
            Voltage[crank_idx] = np.random.normal(34.0, 1.2, len(crank_idx))

        # ====================================================================
        # --- HARDWARE & ADC CLIPPING LAYER ---
        # Simulates physical gauge limits to prevent mathematically impossible outputs.
        # This MUST run after anomalies so we don't accidentally clip out the failures.
        # ====================================================================
        # ====================================================================
        # --- HARDWARE & ADC CLIPPING LAYER ---
        # Simulates physical gauge limits to prevent mathematically impossible outputs.
        # ====================================================================
        
        # Analog gauges rest at 50°C and max out at 130°C
        T_oil = np.clip(T_oil, 50.0, 125.0)    
        T_cool = np.clip(T_cool, 50.0, 120.0)  
        
        P_oil = np.clip(P_oil, 0.0, 15.0)                
        P_gear = np.clip(P_gear, 0.0, 25.0)              
        RPM = np.clip(RPM, 0.0, 2200.0)                  
        Current = np.clip(Current, -1000.0, 250.0)       
        Voltage = np.clip(Voltage, 8.0, 55.0)            
        Fuel_Percent = np.clip(Fuel_Percent, 0.0, 100.0)

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
# Export 2: Raw Telemetry (CSV and XML)
print("\n[SYSTEM] Compiling tabular telemetry...")
df_telemetry = pd.concat(dataset_raw_telemetry, ignore_index=True)

print("[SYSTEM] Exporting to CSV...")
df_telemetry.to_csv("t72_raw_telemetry.csv", index=False)

print("[SYSTEM] Exporting to XML (This may take a moment for large datasets)...")
#df_telemetry.to_xml("t72_raw_telemetry.xml", index=False, root_name="FleetTelemetry", row_name="SensorReading", parser="etree")

# --- EXECUTION SUMMARY ---
print("\n========================================================")
print("  DATA GENERATION SUMMARY")
print("========================================================")
print(f"  Total Data Points (Rows)  : {total_data_points:,}")
print(f"  Tabular Dimensions        : {df_telemetry.shape}")
print("--------------------------------------------------------")
for unit, count in points_per_engine.items():
    print(f"  -> Engine {unit} generated      : {count:,} points")
print("========================================================\n")
print("[SYSTEM] Export Complete. Saved to 't72_ml_features.csv' and 't72_raw_telemetry.xml'.")