# Hydro_Sim2Real_Aquifer_Inversion
Simulation-to-Reality Transfer Learning for Groundwater Aquifer Parameter Inversion
## Project Background
Aiming at the problem of insufficient monitoring data in groundwater parameter inversion, low precision of traditional methods.
This project adopts Sim2Real transfer learning framework: pre-training with synthetic simulation data, fine-tuning with limited real measured groundwater data.

## File Structure
- 01_data: Original real groundwater monitoring data
- 02_code: Core project running scripts
- 03_model_weights: Trained model weights and data scaler files
- 04_results: Model prediction results and output figures
- 05_env: Project operating environment configuration files
- 06_archive: Historical redundant backup files

## Environment Setup
1. Conda environment build