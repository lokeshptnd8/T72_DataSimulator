# T-72 V-46-6 Digital Twin & Synthetic Data Simulator

## Overview
This repository contains a physics-informed digital twin simulator for the V-46-6 heavy diesel engine. It utilizes a thermodynamic and fluid dynamic ODE solver to generate synthetic, run-to-failure time-series telemetry.

The simulator bridges the gap between theoretical mechanical degradation and edge-AI deployment. It generates 9-channel telemetry mapped to strict hardware pinouts, designed to train low-resource predictive maintenance architectures (e.g., MixLinear, MT-GRU).

## Outputs
The pipeline generates two distinct artifacts:
1. **Raw SCADA Telemetry (`.npz`)**: A 1-Hz time-series matrix simulating the physical edge-device hardware across the fleet's lifespan.
2. **ML Feature Targets (`.csv`)**: Extracted degradation targets (e.g., Thermal Soak Rate, Warm-Idle Oil Pressure) for Remaining Useful Life (RUL) regression.

## Setup Instructions

**1. Clone the repository and navigate to the directory:**
```cmd
git clone <your-github-repo-url>
cd t72-digital-twin