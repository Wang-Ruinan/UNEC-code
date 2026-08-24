# UNEC-code
simulation code

This repository contains the code used for the simulation studies. The simulation scripts are named using the convention CS_METHOD_Scenario, where METHOD indicates the selection method, including p-values (Jin& Candès, 2023), Neyman-Pearson (Qin et al. 2025), and UNEC, and Scenario represents the simulation setting.

Each script generally follows the same workflow:

1. Load the required packages.
2. Define the functions used in the simulation.
3. Perform variable selection using different methods.
4. Generate plots to visualize the simulation results.
5. Compute and report the False Discovery Rate (FDR) and Power for evaluating the performance of each method.
