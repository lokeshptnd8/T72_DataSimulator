# ⚙️ T-72 V-46-6 Digital Twin & Synthetic Data Simulator: Master Usage Guide

## 📌 Executive Overview
This repository contains a physics-informed digital twin simulator for the V-46-6 heavy diesel engine. Utilizing a thermodynamic and fluid dynamic ODE (Ordinary Differential Equation) solver, it generates synthetic, run-to-failure time-series telemetry.

The simulator bridges the gap between theoretical mechanical degradation and edge-AI deployment. It generates precise telemetry mapped to strict hardware pinouts, designed specifically to train ultra-low-resource predictive maintenance architectures for environments with severe compute and VRAM constraints.

## 🏗️ Core Architecture

### 1. The Physics Engine
The core solver (`engine_core.py`) mathematically models thermal soak rates, bearing clearance expansions, gearbox friction multipliers, and 48V battery sulfation internal resistance. 

### 2. Hardware-Aligned Telemetry
The pipeline outputs a 1-Hz time-series database matrix structured exactly to the physical edge-device hardware telemetry unit:
* **CH1**: Engine Oil Temperature (°C)
* **CH2**: Coolant Temperature (°C)
* **CH3**: Engine Oil Pressure ($kg/cm^2$)
* **CH4**: Gearbox Oil Pressure ($kg/cm^2$)
* **CH5**: Engine Speed (RPM)
* **CH6**: Battery Current (A)
* **CH7**: Battery Voltage (V)
* **CH8**: Fuel Level (%)
* **CH9**: Engine Hour Meter (Cumulative)

### 3. Target ML Topologies (Phase 0/1)
The synthetic data is engineered to feed low-footprint inference models:
* **MixLinear**: An extreme low-resource multivariate time-series forecasting model requiring only ~0.1K parameters, replacing heavy LSTMs to track monotonic degradation slopes.
* **MT-GRU (Multi-Task Gated Recurrent Unit)**: A shared-representation recurrent network that simultaneously outputs the Remaining Useful Life (RUL) regression curve and categorical Anomaly Classifications while demanding <50MB of RAM.

---

## 🚀 Phase 1: Environment Setup & Installation

Before running any simulations, you must initialize the isolated Python environment and install the required thermodynamic and data-processing libraries.

**1. Clone the Repository & Navigate to the Directory**
Open your terminal and run:
`git clone <your-github-repo-url>`
`cd t72-digital-twin`

**2. Create the Virtual Environment**
Isolate your dependencies to prevent conflicts:
`python -m venv venv`

**3. Activate the Virtual Environment**
`venv\Scripts\activate`

**4. Install Required Libraries**
`pip install -r requirements.txt`

---

## 💻 Phase 2: Running the Simulators

The architecture supports two distinct data generation workflows depending on your engineering objective.

### Workflow A: Environmental Fleet Simulation
Use this workflow to generate synthetic datasets from Hour 0 to End-of-Life across a fleet of virtual engines. This workflow utilizes the Command Line Interface (CLI) to dynamically inject global weather conditions into the physics ODE solver.

**Basic Execution (Standard Lab Conditions: 25°C, 40% Humidity):**
`python final_generator.py`

**Custom Execution (Extreme Weather Testing):**
Pass custom environmental parameters to evaluate how high heat or humidity impacts the radiator cooling efficiency and thermal soak rates.
`python final_generator.py --temp 45.0 --humidity 85.0`
* `--temp`: Sets the ambient external temperature in Celsius.
* `--humidity`: Sets the external relative humidity percentage.

### Workflow B: Hybrid SD Card Calibration
Use this workflow to eliminate domain shift. This script ingests real-world telemetry from the physical tank to reverse-engineer its current wear state and exact sensor noise variance before forecasting its future degradation.

**Prerequisite:** Ensure a file named `sd_card_1hr_log.csv` (containing the 9-channel hardware pinout telemetry for 1 hour of operation) is placed in the root directory.

**Execution:**
`python calibrated_simulator.py`

---

## 📂 Phase 3: Accessing the Outputs

Upon successful execution, the simulators will generate artifacts locally in your directory. These are intentionally ignored by `.gitignore` to prevent repository bloat.

**From Workflow A (`final_generator.py`):**
1. `t72_raw_9channel_telemetry.npz`: The heavy, 1-Hz SCADA matrix `[Sessions, Time, 9-Channels]`. Use this file to populate your local SQLite database for the edge UI.
2. `t72_ml_features.csv`: The lightweight, extracted degradation targets (Warm-Idle Oil Pressure, Thermal Soak Rate, etc.). Use this file directly to train the MixLinear or XGBoost RUL models.

**From Workflow B (`calibrated_simulator.py`):**
1. `calibrated_future_forecast.csv`: The forecasted future telemetry trajectory, mathematically anchored to the specific physical tank's baseline wear and electrical noise profile.

---

## 🧹 Maintenance Commands

When you are finished generating data and wish to exit the project environment, simply deactivate the virtual workspace:
`deactivate`