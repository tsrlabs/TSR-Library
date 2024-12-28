import os
import json
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Directory setup
results_dir = "simulation_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Constants
c = 3.0e8  # Speed of light
k_B = 1.38e-23  # Boltzmann constant
T_initial = 300  # Initial temperature in Kelvin
alpha = 1.0  # Proportional constant for TSR

# Quark masses in GeV/c^2
quark_masses = {
    'up': 2.3e-3,  
    'down': 4.8e-3,
    'charm': 1.28,
    'strange': 0.095,
    'top': 173.0,
    'bottom': 4.18
}

# Convert quark masses to kg
quark_masses_kg = {key: value * 1.782662e-27 for key, value in quark_masses.items()}

# Number of particles
num_particles = 1000
dt = 1e-3
num_steps = 1000
num_runs = 10

# Logging setup
logging.basicConfig(filename='simulation.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

def log_simulation_run(params, results):
    logging.info(f"Parameters: {params}")
    logging.info(f"Results: {results}")

# Arrays to store results
correlation_results = cp.zeros((num_runs, num_steps))
temperature_results = cp.zeros((num_runs, num_steps, num_particles))

def update_particles_with_tsr(positions, velocities, dt, alpha, masses):
    temperatures = alpha * cp.square(cp.linalg.norm(velocities, axis=1)) * masses + 0.01 * cp.random.randn(num_particles)
    velocities += (-k_B * temperatures[:, cp.newaxis] + 0.01 * cp.random.randn(num_particles, 3)) * dt
    positions += velocities * dt
    return positions, velocities, temperatures

def correlation_function(positions):
    dist_matrix = cp.linalg.norm(positions[:, cp.newaxis] - positions, axis=2)
    correlation = cp.mean(cp.exp(-dist_matrix))
    return correlation

params = {'num_particles': num_particles, 'alpha': alpha, 'dt': dt, 'num_steps': num_steps}
logging.info(f"Initializing simulation with {params}")

for run in tqdm(range(num_runs), desc="Overall Progress"):
    # Randomly select quark masses for particles
    masses = np.random.choice(list(quark_masses_kg.values()), num_particles)
    masses = cp.asarray(masses)
    
    positions = cp.random.rand(num_particles, 3) + 0.01 * cp.random.randn(num_particles, 3)
    velocities = cp.random.rand(num_particles, 3) + 0.01 * cp.random.randn(num_particles, 3)
    
    for step in range(num_steps):
        dt = 1e-3 + 1e-5 * cp.random.randn()
        positions, velocities, temperatures = update_particles_with_tsr(positions, velocities, dt, alpha, masses)
        correlation_results[run, step] = correlation_function(positions)
        temperature_results[run, step] = temperatures

    results = {'run': run, 'positions': positions.tolist(), 'velocities': velocities.tolist(), 'temperatures': temperatures.tolist()}
    log_simulation_run(params, results)

def save_results_to_file(run, correlation_results, temperature_results):
    filename = os.path.join(results_dir, f'simulation_results_run_{run}.json')
    with open(filename, 'w') as f:
        json.dump({'correlation_results': correlation_results.tolist(), 'temperature_results': temperature_results.tolist()}, f)

correlation_results_np = cp.asnumpy(correlation_results)
temperature_results_np = cp.asnumpy(temperature_results)

for run in range(num_runs):
    save_results_to_file(run, correlation_results_np[run], temperature_results_np[run])

average_correlation = np.mean(correlation_results_np, axis=0)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(num_steps), average_correlation, label="Average Correlation")
plt.xlabel("Time Steps")
plt.ylabel("Correlation")
plt.title("Average Correlation Over Time with TSR")
plt.legend()

average_temperature = np.mean(temperature_results_np, axis=(0, 2))
plt.subplot(1, 2, 2)
plt.plot(range(num_steps), average_temperature, label="Average Temperature", color='orange')
plt.xlabel("Time Steps")
plt.ylabel("Temperature (K)")
plt.title("Average Temperature Over Time with TSR")
plt.legend()

plt.tight_layout()
plt.show()

