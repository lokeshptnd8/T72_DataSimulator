# ⚙️ T-72 V-46-6 Synthetic Data Simulator

## 📌 Executive Overview
This repository contains a physics-informed digital twin simulator for the V-46-6 heavy diesel engine. Utilizing a thermodynamic and fluid dynamic ODE (Ordinary Differential Equation) solver, it generates synthetic, run-to-failure time-series telemetry.

The simulator bridges the gap between theoretical mechanical degradation and edge-AI deployment. It generates precise telemetry mapped to strict hardware pinouts, designed specifically to train ultra-low-resource predictive maintenance architectures engineered for environments with severe compute and VRAM constraints.

## 🏗️ Core Architecture (Phase 0 Prototype)

### 1. The Physics Engine
The core solver (`engine_core.py`) mathematically models thermal soak rates, bearing clearance expansions, gearbox friction multipliers, and 48V battery sulfation internal resistance. To accommodate the Phase 0 constraint of lacking long-term fleet data, the baseline degradation relies on robust physics-rate linear slope-to-threshold extrapolation.

### 2. Hardware-Aligned Telemetry
The pipeline outputs a 1-Hz time-series matrix structured exactly to the physical edge-device hardware telemetry unit:
* **CH1**: Engine Oil Temperature (°C)
* **CH2**: Coolant Temperature (°C)
* **CH3**: Engine Oil Pressure ($kg/cm^2$)
* **CH4**: Gearbox Oil Pressure ($kg/cm^2$)
* **CH5**: Engine Speed (RPM)
* **CH6**: Battery Current (A)
* **CH7**: Battery Voltage (V)
* **CH8**: Fuel Level (%)
* **CH9**: Engine Hour Meter (Cumulative)

### 3. Target ML Topologies & Infrastructure
To satisfy extreme edge-compute constraints, this pipeline relies on highly optimized downstream targets:
* **MixLinear**: An extreme low-resource multivariate time-series forecasting model requiring only ~0.1K parameters.
* **MT-GRU (Multi-Task Gated Recurrent Unit)**: A shared-representation recurrent network for simultaneous RUL regression and Anomaly Classification requiring <50MB of RAM.
* **Edge Database**: The time-series telemetry is designed to be unified into a local **SQLite** database, intentionally avoiding JVM-heavy time-series stores (like QuestDB) to preserve up to 2 GB of system RAM.
* **Graph Store**: Utilizes an in-memory NetworkX graph loaded from JSON, bypassing the need for a full PostgreSQL server footprint.

---

## 🚀 Phase 1: Environment Setup & Installation

Before running any simulations, initialize the isolated Python environment and install the required thermodynamic libraries.

**1. Clone the Repository & Navigate to the Directory**
```cmd
git clone <your-github-repo-url>
cd t72-digital-twin
```

**2. Create the Virtual Environment**
```cmd
python -m venv venv
```

**3. Activate the Virtual Environment**
```cmd
venv\Scripts\activate
```

**4. Install Required Libraries**
```cmd
pip install -r requirements.txt
```

---

## 💻 Phase 2: Running the Simulators

Both simulation pipelines feature an **Interactive CLI Wizard**. You do not need to pass complex command-line arguments; simply execute the script and follow the on-screen prompts to dynamically shape your fleet size, weather conditions, and session durations. 

*(Note: Press `ENTER` on any prompt to automatically use the standard engineering default).*

### Workflow A: Interactive Fleet Simulation (`final_generator.py`)
Use this workflow to generate massive synthetic datasets from Hour 0 to End-of-Life across a completely virtual fleet.

**Execution:**
```cmd
python final_generator.py
```

### Workflow B: Hybrid SD Card Calibration (`calibrated_simulator.py`)
Use this workflow to eliminate domain shift. This script ingests real-world telemetry from the physical tank (`sd_card_1hr_log.csv`) to extract the exact sensor noise variance and reverse-engineer the tank's current physical wear state before extrapolating its future degradation. 

*(Note: If the physical SD card file is missing, the script will automatically generate a mock hardware baseline to ensure the simulation still executes seamlessly).*

**Execution:**
```cmd
python calibrated_simulator.py
```

---

## 📂 Phase 3: Accessing Outputs & Data Provenance

Upon execution, the terminal will provide a **Data Generation Summary**, detailing the exact number of rows calculated by the ODE solver and the final 3D tensor shape.

The simulators generate artifacts locally in your directory (intentionally ignored by `.gitignore` to prevent repository bloat):

1. **`t72_raw_9channel_telemetry.npz` / `t72_calibrated_telemetry.npz`**: The heavy, 1-Hz SCADA matrix (Formatted as `[Sessions, Time_Steps, 9_Channels]`).
2. **`t72_ml_features.csv`**: The lightweight, extracted degradation targets directly formatted to train Phase 1 RUL inference models.

---

## 🧹 Maintenance Commands

When you are finished generating data and wish to exit the project environment, deactivate the virtual workspace:
```cmd
deactivate
```