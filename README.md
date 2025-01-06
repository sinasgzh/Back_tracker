# Particle Tracking in Plasma Sheet

This project implements a simulation to track the motion of charged particles (electrons and protons) in dynamically varying electromagnetic fields. It uses field data from RCM runs, interpolates it onto a common grid, and applies Runge-Kutta 4th order (RK4) integration to compute particle trajectories. The results are saved and visualized using Python libraries.

# Features
## Field Data Loading and Preprocessing
Reads field data files (datapot.dat) and splits them into chunks based on time intervals.
Interpolates fields onto a common grid for consistent analysis.
## Particle Initialization
Generates initial positions for electrons and protons within specified radial and angular ranges.
Ensures particles are placed with a minimum distance constraint if enabled.
## Velocity Computation
Computes total velocities (V_tot) for particles by combining gradient-curvature drift and ExB drift velocities.
## Trajectory Calculation
Tracks particles using RK4 integration. Stops particles when they exit a predefined elliptical boundary.
## Visualization
Plots PV_gamma maps overlaid with particle positions and velocities. Highlights particles with color-coded markers based on their energy (landa values).
## Data Storage
Saves particle tracking results and field properties in an HDF5 file (tracking_results.h5).
# Installation
## Prerequisites
Python 3.8+
## Libraries
numpy
pandas
matplotlib
scipy
h5py
## Install the required libraries using pip
pip install numpy pandas matplotlib scipy h5py
# Usage
## Run the Simulation
## Execute the script directly
python particle_tracking.py
## Input Files
Ensure the field data file (datapot.dat) is placed in the directory:
/content/drive/MyDrive/ColabNotebooks/RCM_out/
## Output
The processed data, particle trajectories, and plots are saved in the directory:
/content/drive/MyDrive/ColabNotebooks/BT_out/
# Code Overview
## Key Functions
**generate_initial_positions:**
Generates random positions for particles within specified ranges.

**load_data and split_data:**
Reads field data and splits it into time-based chunks.

**interpolate_fields:**
Interpolates field data onto a common grid.

**calculate_vtot**:
Computes total velocities based on field gradients and drift physics.

**rk4_step**:
Advances particle positions using RK4 integration.

**process_particles:**
Tracks particle trajectories through the field.

**plot_PV_gamma_map_with_particles:**
Visualizes PV_gamma maps with particle positions and velocities.

**advance_particle:**
Handles particle boundary conditions and motion calculations.

## Example Output
Trajectory Plots:
Plots of PV_gamma maps with overlaid electron and proton trajectories.
HDF5 Data:
Particle positions, velocities, and associated field values for each chunk.
## Customization
**Adjust Particle Parameters:**
Modify the num_electrons, num_protons, R1, R2, phi1, and phi2 parameters in the generate_initial_positions function.

**Change Field Resolution:**
Update grid_resolution in the interpolate_fields function.

**Enable/Disable Drift Effects:**
Use the include_gc flag to include/exclude gradient-curvature drift calculations.

_______________________________

**Author**

Sina Sadeghzadeh

**Feel free to contact me via sinasgzh@gmail.com for any questions or suggestions.**
