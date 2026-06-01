# config.py
# Physical constants for the V-46-6 Healthy Baseline Engine (FINE-TUNED)

PHYSICS = {
    # Thermal Mass 
    "C_cool": 800000.0,  
    "C_oil": 500000.0,   
    
    # Heat Transfer Coefficients (BALANCED)
    "h_rad": 1800.0,    # Reduced so the coolant is allowed to warm up to 85C
    "h_oc": 3500.0,     # Massively increased so oil dumps its heat into the coolant
    "h_oa": 400.0,       
    
    # Lubrication & Fluid Dynamics
    "mu_0": 0.1,        
    "visc_b": 0.005,    
    "k_pump": 0.075,    # Dialed down to yield ~5.8 kg/cm2 at warm idle
    "P_relief": 8.5,    
    "C_clear_0": 1.0,   
    
    # Rotational & Mechanical
    "J_engine": 15.0,   
    "k_friction": 0.05, 
    
    # Environment
    "T_ambient": 25.0,   

    # Fuel & Electrical Block ---
    "fuel_capacity_0": 1200.0, # liters (Internal + external tanks)
    "fuel_burn_rate": 0.00015, # Volumetric depletion multiplier
    
    "V_battery_0": 24.0,       # Nominal battery voltage (Engine Off)
    "V_alt_reg": 28.5,         # Alternator voltage regulator setpoint
    "alt_pulley_ratio": 3.0    # Alternator spins 3x faster than engine
}