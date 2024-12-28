import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import pandas as pd
import json
import os

# Constants
c = 299792458  # Speed of light in m/s
E_mc2 = c**2  # Mass-energy equivalence in J/kg
TSR = E_mc2 / (1.38e-23)  # Temperature to Speed Ratio in K/m/s
alpha = 1.0  # Proportional constant for TSR
Q = 2 ** (1/12)  # Fractal structure parameter
dark_energy_density = 5.96e-27  # Density of dark energy in kg/m^3
dark_matter_density = 2.25e-27  # Density of dark matter in kg/m^3
collision_distance = 1e-10  # Distance for collision detection
Hubble_constant = 70.0  # km/s/Mpc (approximation)
Hubble_constant_SI = Hubble_constant * 1000 / 3.086e22  # Hubble constant in SI units (s^-1)

# Initial conditions
temperature_initial = 1.42e32  # Planck temperature in K
particle_density_initial = 5.16e96  # Planck density in kg/m^3
particle_speed_initial = c * 0.999  # Slightly less than c initially

# Simulation time
t_planck = 5.39e-44  # Planck time in s
t_simulation = t_planck * 1e3  # Shorter timescale for simulation

# Quark masses (in GeV)
quark_masses = {
    'up': 2.3e-3,
    'down': 4.8e-3,
    'charm': 1.28,
    'strange': 0.095,
    'top': 173.0,
    'bottom': 4.18
}

# Conversion factor from GeV to J
GeV_to_J = 1.60217662e-10

# Simulation setup
num_steps = int(t_simulation / t_planck)
num_quarks = len(quark_masses)

# Tunneling probabilities to investigate
tunneling_probabilities = np.arange(0.1, 1.1, 0.1)

# Create a directory to store the data
data_dir = 'big_bang_simulation_data'
os.makedirs(data_dir, exist_ok=True)

# Functions
def relativistic_energy(particle_speed, particle_mass):
    if particle_speed >= c:
        return particle_mass * c**2  # Handle particle_speed >= c
    return particle_mass * c**2 / np.sqrt(1 - (particle_speed / c)**2)

def relativistic_momentum(particle_speed, particle_mass):
    if particle_speed >= c:
        return np.inf  # Handle particle_speed >= c cases appropriately.
    return particle_mass * particle_speed / np.sqrt(1 - (particle_speed / c)**2)  # More precise calculation

def update_speed(current_speed, current_temperature, particle_mass):
    rel_momentum = relativistic_momentum(current_speed, particle_mass)
    return c * np.sqrt(max(0, 1 - (rel_momentum / (rel_momentum + dark_energy_density))**2))

def run_simulation(tunneling_probability):
    particle_speeds = np.zeros((num_quarks, num_steps))
    particle_temperatures = np.zeros((num_quarks, num_steps))
    particle_masses_evolution = np.zeros((num_quarks, num_steps))
    particle_positions = np.zeros((num_quarks, num_steps))
    tunneling_steps = np.zeros((num_quarks, num_steps), dtype=bool)
    particle_masses = np.array(list(quark_masses.values())) * GeV_to_J  # Convert quark masses to Joules
    particle_speeds[:, 0] = particle_speed_initial
    particle_temperatures[:, 0] = temperature_initial
    particle_masses_evolution[:, 0] = particle_masses
    particle_positions[:, 0] = np.zeros(num_quarks)  # Initial positions

    for step in range(1, num_steps):
        for i in range(num_quarks):
            # Update temperature based on some physical model (placeholder)
            particle_temperatures[i, step] = particle_temperatures[i, step - 1] * (1 - 0.01)  # Cooling down

            # Update speed based on temperature and mass
            particle_speeds[i, step] = update_speed(particle_speeds[i, step - 1], particle_temperatures[i, step], particle_masses[i])

            # Simulate tunneling effect
            if np.random.rand() < tunneling_probability:
                tunneling_steps[i, step] = True
                # Modify mass or speed based on tunneling (placeholder)
                particle_masses[i] *= 0.9  # Example: reduce mass by 10%
                particle_speeds[i, step] *= 1.1  # Example: increase speed by 10%

            # Store the current mass
            particle_masses_evolution[i, step] = particle_masses[i]

            # Update position based on speed
            particle_positions[i, step] = particle_positions[i, step - 1] + particle_speeds[i, step] * t_planck

    return particle_positions, particle_speeds, particle_temperatures, tunneling_steps, particle_masses_evolution

# Run the simulation for each tunneling probability and save the results
for tunneling_probability in tunneling_probabilities:
    positions, speeds, temperatures, tunneling_steps, masses = run_simulation(tunneling_probability)

    # Calculate the correlation matrix for speeds, temperatures, and masses
    data_for_correlation = np.vstack((speeds.flatten(), temperatures.flatten(), masses.flatten())).T
    correlation_matrix = np.corrcoef(data_for_correlation, rowvar=False)

    # Save the results to a JSON file
    results = {
        'tunneling_probability': tunneling_probability,
        'positions': positions.tolist(),
        'speeds': speeds.tolist(),
        'temperatures': temperatures.tolist(),
        'tunneling_steps': tunneling_steps.tolist(),
        'masses': masses.tolist(),
        'correlation_matrix': correlation_matrix.tolist()  # Save the correlation matrix
    }

    with open(os.path.join(data_dir, f'results_tunneling_{tunneling_probability:.1f}.json'), 'w') as f:
        json.dump(results, f)

# Visualization (optional)
def plot_results(positions, speeds, temperatures, tunneling_steps, masses):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_quarks):
        ax.plot(positions[i], speeds[i], temperatures[i], label=f'Quark {i+1}')

    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_zlabel('Temperature (K)')
    ax.set_title('Big Bang Simulation Results')
    ax.legend()
    plt.show()

# Example of plotting results for the first tunneling probability
with open(os.path.join(data_dir, 'results_tunneling_0.1.json'), 'r') as f:
    data = json.load(f)
    plot_results(np.array(data['positions']), np.array(data['speeds']), np.array(data['temperatures']), np.array(data['tunneling_steps']), np.array

