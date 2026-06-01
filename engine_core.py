# engine_core.py
import numpy as np
from config import PHYSICS as p

def engine_derivatives(t, state, rpm_req, load_req):
    """
    Expanded ODEs: state = [T_cool, T_oil, omega, V_fuel]
    """
    T_cool, T_oil, omega, V_fuel = state
    
    error = rpm_req - omega
    torque_engine = 5.0 * error  
    if torque_engine < 0: torque_engine = 0 
    
    torque_load = load_req * omega * 0.1 
    torque_friction = p["k_friction"] * omega
    d_omega_dt = (torque_engine - torque_load - torque_friction) / p["J_engine"]
    
    Q_comb = torque_engine * omega * 0.3 
    Q_friction = torque_friction * omega * 0.8
    
    # Thermal Block
    Q_rad = p["h_rad"] * (T_cool - p["T_ambient"]) 
    Q_oc = p["h_oc"] * (T_oil - T_cool)            
    Q_oa = p["h_oa"] * (T_oil - p["T_ambient"])    
    
    d_Tcool_dt = (Q_comb * 0.4 - Q_rad + Q_oc) / p["C_cool"]
    d_Toil_dt = (Q_friction + Q_comb * 0.1 - Q_oc - Q_oa) / p["C_oil"]
    
    # NEW: Fuel Block (Depletion based on combustion heat generation)
    d_Vfuel_dt = -p["fuel_burn_rate"] * Q_comb 
    if V_fuel <= 0: d_Vfuel_dt = 0 # Cannot burn fuel if tank is empty
    
    return [d_Tcool_dt, d_Toil_dt, d_omega_dt, d_Vfuel_dt]

def calculate_oil_pressure(omega, T_oil, clearance):
    """Calculates oil pressure based on RPM, temperature, and a Pressure Relief Valve."""
    viscosity = p["mu_0"] * np.exp(-p["visc_b"] * (T_oil - 20.0))
    raw_pressure = (p["k_pump"] * omega * viscosity) / (clearance ** 2)
    
    # Simulate the mechanical Pressure Relief Valve (PRV) popping open
    regulated_pressure = np.minimum(raw_pressure, p["P_relief"])
    return regulated_pressure

def calculate_gearoil_pressure(omega, clutch_wear):
    """Transmission RSF Target: Nominal is 11.2 kg/cm2 at 1600-1900 RPM."""
    rpm = omega # REMOVED 9.549 multiplier
    base_pressure = np.minimum(11.5, (rpm / 1600.0) * 11.5)
    
    pressure = base_pressure * clutch_wear
    return pressure

def calculate_electrical(omega, battery_sulfation):
    """Battery GPR Target: 48V cranking transient, dipping lower as sulfation increases."""
    rpm = omega # REMOVED 9.549 multiplier
    alt_freq = rpm * p["alt_pulley_ratio"] * 0.1 
    
    is_cranking = (rpm > 0) & (rpm < 400)
    is_running = (rpm >= 400)
    
    raw_voltage = np.where(is_running, p["V_alt_reg"], 
                  np.where(is_cranking, 48.0, p["V_battery_0"]))
    
    voltage = np.where(is_cranking, raw_voltage - battery_sulfation, raw_voltage)
    current = np.where(is_running, 15.0 - (rpm * 0.001), 
              np.where(is_cranking, -400.0, -10.0)) 
    
    return alt_freq, voltage, current