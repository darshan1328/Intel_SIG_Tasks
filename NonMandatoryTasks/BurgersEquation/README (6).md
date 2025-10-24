# 2D Burgers' Equation Synthetic Data Generator

## Overview

This repository contains a Notebook (`nmt-burgers-equation.ipynb`) designed to generate and visualize synthetic datasets for the 2D Burgers' equation.

The primary purpose of this script is to create data based on a known analytical solution to the equation. This data can then be used for tasks such as training, validating, or testing machine learning models.



## Functionality

The notebook is structured into three main parts: data generation, parameter setup, and visualization.

### 1. Data Generation

The data is generated using a specific analytical solution to the 2D Burgers' equation, defined in the `burgers_2d_analytical` function:

* **U-velocity (u):** `u = sin(π * x) * sin(π * y) * exp(-2 * ν * π² * t)`
* **V-velocity (v):** `v = cos(π * x) * cos(π * y) * exp(-2 * ν * π² * t)`

The `generate_burgers_2d_dataset` function takes a list of viscosity values (`nu_values`) and grid parameters (nx, ny, nt) to compute the U and V velocities over the complete spatio-temporal domain.

### 2. Visualization

Two helper functions are provided to visualize the generated data:

* **`visualize_burgers_patterns`**: This function generates a series of plots for each viscosity value provided. For each `nu`, it displays:
    * A contour plot of the U velocity field at a midpoint in time.
    * A contour plot of the V velocity field at a midpoint in time.
    * A contour plot of the velocity magnitude with a quiver plot overlay to show vector direction.
    This function also saves the combined plots to `burgers_patterns.png`.

<img width="1136" height="377" alt="Screenshot 2025-10-24 182415" src="https://github.com/user-attachments/assets/7de66243-6218-4244-a1ce-d67640f23b64" />

<img width="1151" height="491" alt="Screenshot 2025-10-24 182509" src="https://github.com/user-attachments/assets/5e3408b0-ea11-4f2f-83e5-44c49fe4378b" />

<img width="1130" height="478" alt="Screenshot 2025-10-24 182501" src="https://github.com/user-attachments/assets/84cc1865-cae7-4850-9c29-e144d06b5687" />


* **`visualize_time_evolution`**: This function takes the data for a *single* viscosity value and generates a series of contour plots for both U and V velocities at different time steps (t=0, 0.25, 0.5, 0.75, 1.0), illustrating the decay of the velocity fields over time.

<img width="1148" height="458" alt="Screenshot 2025-10-24 182445" src="https://github.com/user-attachments/assets/900b159b-cffc-4c52-94a9-5a5c8b2fbc7c" />


## Model characteristics:

- Input: Four features — spatial coordinates (x, y), time t, and viscosity ν

- Output: Two targets — velocity components (u, v)

- Hidden layers: Multiple dense layers with ReLU or Tanh activations

- Loss: Mean Squared Error (MSE)

- Optimizer: Adam with adaptive learning rate

Training minimizes the prediction error between the analytical and neural outputs through standard backpropagation. Performance is validated using unseen data and visual error analysis.

### Key Features:
- Synthetic 2D Burgers' Dataset Generation

- Time-series data with varying viscosity ν
- Analytical solution using traveling wave approach


- Pattern analysis for different ν values

- Train and test use DIFFERENT ν values

### PINN Architecture

- Feed-forward neural network with customizable layers
- Automatic differentiation for computing PDE residuals
- Combined data and physics loss
- Noise Analysis

- Compares clean vs noisy training

Detailed Evaluation

- True vs predicted comparisons
- Error visualization
- Quantitative metrics (MSE, MAE)

## Dependencies

This script requires Python 3 and the following core libraries:

* **NumPy**: For numerical operations and array manipulation.
* **PyTorch**: Used in this notebook for setting the random seed to ensure reproducibility.
* **Matplotlib**: For generating the 2D contour and quiver plots.
* **Seaborn**: For enhancing the plot aesthetics.
* **SciPy**: The `odeint` function is imported but not utilized in the final analytical solution provided.


