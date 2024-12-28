import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import cupy as cp
import logging
import os
import json
from tqdm import tqdm


# Create results directory
results_dir = "quantum_convergence_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Logging setup
logging.basicConfig(filename=os.path.join(results_dir, 'simulation.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

def log_simulation(params, results):
    logging.info(f"Parameters: {params}")
    logging.info(f"Results: {results}")

# Constants
c = 3.0e8  # Speed of light
h = 6.626e-34  # Planck's constant
k_B = 1.38e-23  # Boltzmann constant
Q = 2 ** (1/12)  # Twelfth root of 2

# Simulation parameters
num_runs = 500
num_points = 100
num_particles = 1000
dt = 1e-3
num_steps = 1000
alpha = 1.0  # Proportional constant for TSR

# Result arrays
m_A_results = np.zeros((num_runs, num_points))
m_B_results = np.zeros((num_runs, num_points))
correlation_results_combined = np.zeros((num_runs, num_points))
correlation_results = cp.zeros((num_runs, num_steps))
temperature_results = cp.zeros((num_runs, num_steps, num_particles))

# Functions
def wave_function(x, t, scale=1.0, phase_shift=0.0):
    denominator = 2 * (t**2 + 1e-10)
    return scale * Q * np.exp(-x**2 / denominator) * np.exp(-1j * (t + phase_shift))

def green_function(x, t, x_prime, t_prime, scale=1.0, phase_shift=0.0):
    denominator = 2 * (t**2 + 1e-10)
    return scale * Q * np.exp(-((x - x_prime)**2 + (t - t_prime)**2) / denominator) * np.exp(-1j * (t - t_prime + phase_shift))

def zero_point_energy(x, t, scale=1.0, phase_shift=0.0):
    denominator = 2 * (t**2 + 1e-10)
    return scale * Q * np.exp(-x**2 / denominator) * np.exp(-1j * (t + phase_shift))

def quantum_non_locality(x, t, scale=1.0, phase_shift=0.0):
    denominator = 2 * (t**2 + 1e-10)
    return scale * Q * np.exp(-x**2 / denominator) * np.exp(-1j * (t + phase_shift))

def update_particles_with_tsr(positions, velocities, dt, alpha):
    temperatures = alpha * cp.square(cp.linalg.norm(velocities, axis=1)) + 0.01 * cp.random.randn(num_particles)
    velocities += (-k_B * temperatures[:, cp.newaxis] + 0.01 * cp.random.randn(num_particles, 3)) * dt
    positions += velocities * dt
    return positions, velocities, temperatures

def correlation_function(positions):
    dist_matrix = cp.linalg.norm(positions[:, cp.newaxis] - positions, axis=2)
    correlation = cp.mean(cp.exp(-dist_matrix))
    return correlation

# Meshgrid for wave functions
x = np.linspace(-10, 10, num_points)
t = np.linspace(0, 10, num_points)
X, T = np.meshgrid(x, t)

# Arrays to store wave function results
wave_functions = []
green_functions = []
zero_point_energies = []
quantum_non_localities = []

# Simulate different variants
scales = [0.5, 1.0, 1.5]
phase_shifts = [0, np.pi/4, np.pi/2]

for scale in scales:
    for phase_shift in phase_shifts:
        wave_functions.append(wave_function(X, T, scale, phase_shift))
        green_functions.append(green_function(X, T, X, T, scale, phase_shift))
        zero_point_energies.append(zero_point_energy(X, T, scale, phase_shift))
        quantum_non_localities.append(quantum_non_locality(X, T, scale, phase_shift))

wave_functions = np.array(wave_functions)
green_functions = np.array(green_functions)
zero_point_energies = np.array(zero_point_energies)
quantum_non_localities = np.array(quantum_non_localities)

# Average and standard deviation calculations
avg_wave_function = np.mean(np.abs(wave_functions), axis=0)
std_wave_function = np.std(np.abs(wave_functions), axis=0)
avg_green_function = np.mean(np.abs(green_functions), axis=0)
std_green_function = np.std(np.abs(green_functions), axis=0)
avg_zero_point_energy = np.mean(np.abs(zero_point_energies), axis=0)
std_zero_point_energy = np.std(np.abs(zero_point_energies), axis=0)
avg_quantum_non_locality = np.mean(np.abs(quantum_non_localities), axis=0)
std_quantum_non_locality = np.std(np.abs(quantum_non_localities), axis=0)

# Simulation for Particle A and B
for run in tqdm(range(num_runs), desc="Overall Progress"):
    positions = cp.random.rand(num_particles, 3) + 0.01 * cp.random.randn(num_particles, 3)
    velocities = cp.random.rand(num_particles, 3) + 0.01 * cp.random.randn(num_particles, 3)
    
    # Define params for this run
    params = {
        'run': run,
        'num_particles': num_particles,
        'alpha': alpha,
        'dt': dt,
        # Add any other parameters you want to log
    }
    
    for step in range(num_steps):
        dt = 1e-3 + 1e-5 * cp.random.randn()
        positions, velocities, temperatures = update_particles_with_tsr(positions, velocities, dt, alpha)
        correlation_results[run, step] = correlation_function(positions)
        temperature_results[run, step] = temperatures
    
    T_A = np.linspace(1, 0.01, num_points)  # Temperature range
    E_z = h * 1e12  # Zero-point energy
    m_A = np.zeros_like(T_A)
    for i, T in enumerate(T_A):
        m_A[i] = E_z / (c**2)  # Effective mass calculation
    m_A_results[run, :] = m_A
    
    v_B = np.linspace(0, 0.9999 * c, num_points)  # Speed range
    m_B = np.zeros_like(v_B)
    for i, v in enumerate(v_B):
        gamma = 1 / np.sqrt(1 - (v / c)**2)
        m_B[i] = 1 / gamma  # Effective mass calculation
    m_B_results[run, :] = m_B
    
    correlation_combined = m_A * m_B  # Combined correlation function
    correlation_results_combined[run, :] = correlation_combined

    results = {'run': run, 'positions': positions.tolist(), 'velocities': velocities.tolist(), 'temperatures': temperatures.tolist(),
               'm_A': m_A.tolist(), 'm_B': m_B.tolist(), 'correlation_combined': correlation_combined.tolist()}
    log_simulation(params, results)  # Log the parameters and results


# Save results to files
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

# Plot results
average_correlation = np.mean(correlation_results_np, axis=0)
plt.figure(figsize=(12, 6))

# Average Correlation Plot
plt.subplot(2, 2, 1)
plt.plot(range(num_steps), average_correlation, label="Average Correlation")
plt.xlabel("Time Steps")
plt.ylabel("Correlation")
plt.title("Average Correlation Over Time with TSR")
plt.legend()

# Average Temperature Plot
average_temperature = np.mean(temperature_results_np, axis=0)
plt.subplot(2, 2, 2)
plt.plot(range(num_steps), average_temperature, label="Average Temperature", color='orange')
plt.xlabel("Time Steps")
plt.ylabel("Temperature")
plt.title("Average Temperature Over Time")
plt.legend()

# Average Effective Mass for Particle A Plot
average_m_A = np.mean(m_A_results, axis=0)
plt.subplot(2, 2, 3)
plt.plot(T_A, average_m_A, label="Average Effective Mass of Particle A", color='green')
plt.xlabel("Temperature")
plt.ylabel("Effective Mass (m_A)")
plt.title("Average Effective Mass of Particle A vs Temperature")
plt.legend()

# Average Effective Mass for Particle B Plot
average_m_B = np.mean(m_B_results, axis=0)
plt.subplot(2, 2, 4)
plt.plot(v_B, average_m_B, label="Average Effective Mass of Particle B", color='red')
plt.xlabel("Speed (v_B)")
plt.ylabel("Effective Mass (m_B)")
plt.title("Average Effective Mass of Particle B vs Speed")
plt.legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

