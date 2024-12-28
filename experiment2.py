import os
import logging
import json
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Create results directory
results_dir = "combined_simulation_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Logging setup
logging.basicConfig(filename='combined_simulation.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

def log_simulation_run(params, results):
    logging.info(f"Parameters: {params}")
    logging.info(f"Results: {results}")

# Constants
c = 3.0e8  # Speed of light
h = 6.626e-34  # Planck's constant
k_B = 1.38e-23  # Boltzmann constant
T_initial = 300  # Initial temperature in Kelvin
alpha = 1.0  # Proportional constant for TSR
Q = 2 ** (1/12)  # Twelfth root of 2

# Simulation parameters
num_particles = 1000
dt = 1e-3
num_steps = 1000
num_runs = 10

# Arrays to store results
correlation_results = cp.zeros((num_runs, num_steps))
temperature_results = cp.zeros((num_runs, num_steps, num_particles))
m_A_results = np.zeros((num_runs, 100))
m_B_results = np.zeros((num_runs, 100))
correlation_results_combined = np.zeros((num_runs, 100))

def update_particles_with_tsr(positions, velocities, dt, alpha):
    temperatures = alpha * cp.square(cp.linalg.norm(velocities, axis=1)) + 0.01 * cp.random.randn(num_particles)
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
    positions = cp.random.rand(num_particles, 3) + 0.01 * cp.random.randn(num_particles, 3)
    velocities = cp.random.rand(num_particles, 3) + 0.01 * cp.random.randn(num_particles, 3)
    
    for step in range(num_steps):
        dt = 1e-3 + 1e-5 * cp.random.randn()
        positions, velocities, temperatures = update_particles_with_tsr(positions, velocities, dt, alpha)
        correlation_results[run, step] = correlation_function(positions)
        temperature_results[run, step] = temperatures
    
    T_A = np.linspace(1, 0.01, 100)  # Temperature range
    E_z = h * 1e12  # Zero-point energy
    m_A = np.zeros_like(T_A)
    for i, T in enumerate(T_A):
        m_A[i] = E_z / (c**2)  # Effective mass calculation
    m_A_results[run, :] = m_A
    
    v_B = np.linspace(0, 0.9999 * c, 100)  # Speed range
    m_B = np.zeros_like(v_B)
    for i, v in enumerate(v_B):
        gamma = 1 / np.sqrt(1 - (v / c)**2)
        m_B[i] = 1 / gamma  # Effective mass calculation
    m_B_results[run, :] = m_B
    
    correlation_combined = m_A * m_B  # Combined correlation function
    correlation_results_combined[run, :] = correlation_combined

    results = {'run': run, 'positions': positions.tolist(), 'velocities': velocities.tolist(), 'temperatures': temperatures.tolist(), 
               'm_A': m_A.tolist(), 'm_B': m_B.tolist(), 'correlation_combined': correlation_combined.tolist()}
    log_simulation_run(params, results)

def save_results_to_file(run, correlation_results, temperature_results, m_A_results, m_B_results, correlation_combined):
    filename = os.path.join(results_dir, f'combined_simulation_results_run_{run}.json')
    with open(filename, 'w') as f:
        json.dump({'correlation_results': correlation_results.tolist(), 'temperature_results': temperature_results.tolist(),
                   'm_A_results': m_A_results.tolist(), 'm_B_results': m_B_results.tolist(), 
                   'correlation_combined': correlation_combined.tolist()}, f)

correlation_results_np = cp.asnumpy(correlation_results)
temperature_results_np = cp.asnumpy(temperature_results)

for run in range(num_runs):
    save_results_to_file(run, correlation_results_np[run], temperature_results_np[run], m_A_results[run], m_B_results[run], correlation_results_combined[run])
average_correlation = np.mean(correlation_results_np, axis=0)
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(range(num_steps), average_correlation, label="Average Correlation")
plt.xlabel("Time Steps")
plt.ylabel("Correlation")
plt.title("Average Correlation Over Time with TSR")
plt.legend()

average_temperature = np.mean(temperature_results_np, axis=(0, 2))
plt.subplot(2, 2, 2)
plt.plot(range(num_steps), average_temperature, label="Average Temperature", color='orange')
plt.xlabel("Time Steps")
plt.ylabel("Temperature (K)")
plt.title("Average Temperature Over Time with TSR")
plt.legend()

m_A_mean = np.mean(m_A_results, axis=0)
m_A_std = np.std(m_A_results, axis=0)
plt.subplot(2, 2, 3)
plt.errorbar(T_A, m_A_mean, yerr=m_A_std, fmt='-', ecolor='r', capsize=5)
plt.xlabel('Temperature (K)')
plt.ylabel('Effective Mass (kg)')
plt.title('Particle A: Effective Mass vs Temperature')

m_B_mean = np.mean(m_B_results, axis=0)
m_B_std = np.std(m_B_results, axis=0)
plt.subplot(2, 2, 4)
plt.errorbar(v_B / c, m_B_mean, yerr=m_B_std, fmt='-', ecolor='r', capsize=5)
plt.xlabel('Speed (fraction of c)')
plt.ylabel('Effective Mass (kg)')
plt.title('Particle B: Effective Mass vs Speed')

plt.tight_layout()
plt.show()
