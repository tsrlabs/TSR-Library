import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import pandas as pd

# Constants
c = 299792458  # Speed of light in m/s
E_mc2 = c**2  # Mass-energy equivalence in J/kg
TSR = E_mc2 / (1.38e-23)  # Temperature to Speed Ratio in K/m/s
alpha = 1.0  # Proportional constant for TSR
Q = 2 ** (1/12)  # Fractal structure parameter
dark_energy_density = 5.96e-27  # Density of dark energy in kg/m^3
dark_matter_density = 2.25e-27  # Density of dark matter in kg/m^3
collision_distance = 1e-10  # Distance for collision detection

# Initial conditions
temperature_initial = 1.42e32  # Planck temperature in K
particle_density_initial = 5.16e96  # Planck density in kg/m^3
particle_speed_initial = c  # Initially at the speed of light

# Simulation time
t_planck = 5.39e-44  # Planck time in s
t_simulation = t_planck * 1e3  # Short timescale for simulation

# Quark masses (in GeV) - used for initial mass values and comparison
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
particle_speeds_all = []
particle_temperatures_all = []
particle_masses_evolution_all = []
tunneling_steps_all = []
correlation_matrices = []

# Tunneling probabilities to investigate
tunneling_probabilities = np.arange(0.1, 1.1, 0.1)

# Functions to incorporate relativistic effects
def relativistic_energy(particle_speed, particle_mass):
    if particle_speed >= c:
        return np.inf
    return particle_mass * c**2 / np.sqrt(max(1e-10, 1 - (particle_speed / c)**2))

def relativistic_momentum(particle_speed, particle_mass):
    if particle_speed >= c:
        return np.inf
    return particle_mass * particle_speed / np.sqrt(max(1e-10, 1 - (particle_speed / c)**2))

def update_speed(current_speed, current_temperature, particle_mass):
    rel_momentum = relativistic_momentum(current_speed, particle_mass)
    return c * np.sqrt(max(1e-10, 1 - (rel_momentum / (rel_momentum + dark_energy_density))**2))

# Simulate the Big Bang with Dark Energy, Dark Matter, Tunneling, and Relativistic Effects
for tunneling_probability in tunneling_probabilities:
    print(f"Running simulation for tunneling probability: {tunneling_probability}")
    particle_speeds = np.zeros((len(quark_masses), num_steps))  # 2D array for speeds
    particle_temperatures = np.zeros((len(quark_masses), num_steps))  # 2D array for temperatures
    particle_masses_evolution = np.zeros((len(quark_masses), num_steps))  # 2D array for mass evolution
    particle_positions = np.zeros((len(quark_masses), num_steps))  # 2D array for positions
    tunneling_steps = np.zeros((len(quark_masses), num_steps), dtype=bool)  # 2D array for tunneling steps

    for j, (quark, mass) in enumerate(quark_masses.items()):
        particle_mass = mass * GeV_to_J  # Convert mass to J
        particle_masses_evolution[j, 0] = particle_mass  # Initialize mass
        particle_positions[j, 0] = 0  # Initialize position
        for i in range(1, num_steps):
            particle_speeds[j, i] = update_speed(particle_speeds[j, i - 1], particle_temperatures[j, i - 1], particle_mass)
            particle_positions[j, i] = particle_positions[j, i-1] + particle_speeds[j, i] * t_planck  # Update position
            value = 1 - (particle_speeds[j, i] / (TSR * temperature_initial)) + dark_matter_density
            if np.random.rand() < tunneling_probability:
                particle_speeds[j, i] = particle_speeds[j, 0]  # Tunneling effect
                tunneling_steps[j, i] = True  # Mark tunneling step
            if value < 0:
                value = 0
            particle_temperatures[j, i] = alpha * particle_speeds[j, i]**2  # Apply TSR equation
            # Update mass based on energy conversion
            particle_masses_evolution[j, i] = particle_masses_evolution[j, i-1] * (1 + (particle_speeds[j, i]**2 - particle_speeds[j, i-1]**2) / (2 * c**2))

            # Collision detection and resolution
            for k in range(j+1, len(quark_masses)):
                if abs(particle_positions[j, i] - particle_positions[k, i]) < collision_distance:
                    # Resolve collision (simplified example)
                    particle_speeds[j, i] = -particle_speeds[j, i]
                    particle_speeds[k, i] = -particle_speeds[k, i]
                    # Update temperatures based on TSR
                    particle_temperatures[j, i] = alpha * particle_speeds[j, i]**2
                    particle_temperatures[k, i] = alpha * particle_speeds[k, i]**2

            # Debugging output
            if np.isnan(particle_speeds[j, i]) or np.isnan(particle_temperatures[j, i]):
                print(f"NaN detected at step {i} for quark {quark}")
                print(f"Previous speed: {particle_speeds[j, i-1]}")
                print(f"Previous temperature: {particle_temperatures[j, i-1]}")
                print(f"Current speed: {particle_speeds[j, i]}")
                print(f"Current temperature: {particle_temperatures[j, i]}")
                break

        # Cap speed to avoid unphysical values
        particle_speeds[j] = np.clip(particle_speeds[j], 0, c)

    # Store results for each tunneling probability
    particle_speeds_all.append(particle_speeds)
    particle_temperatures_all.append(particle_temperatures)
    particle_masses_evolution_all.append(particle_masses_evolution)
    tunneling_steps_all.append(tunneling_steps)

    # --- Plotly Interactive Visualization (3D) ---
    # Create the 3D scatter plot using Plotly
    X, Y = np.meshgrid(particle_speeds[0], np.arange(num_steps))  # Create 2D meshgrid for x and y
    fig = go.Figure(data=[go.Scatter3d(x=particle_speeds[j], y=particle_temperatures[j], z=np.arange(num_steps), mode='lines+markers', name=quark.capitalize()) for j, quark in enumerate(quark_masses.keys())])
    fig.update_layout(title=f"Big Bang Simulation: Temperature vs. Speed (Tunneling Probability: {tunneling_probability})", autosize=False, width=800, height=600, margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

    # --- Matplotlib Animation (3D) ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], 'b-')

    # Set axis limits
    ax.set_xlim(min(particle_speeds.flatten()), max(particle_speeds.flatten()))
    ax.set_ylim(min(particle_temperatures.flatten()), max(particle_temperatures.flatten()))
    ax.set_zlim(0, num_steps)

    ax.set_xlabel('Particle Speed')
    ax.set_ylabel('Particle Temperature')
    ax.set_zlabel('Time')
    ax.set_title(f"Big Bang Simulation Animation (Tunneling Probability: {tunneling_probability})")

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def update(frame):
        line.set_data(particle_speeds[:, :frame].flatten(), particle_temperatures[:, :frame].flatten())
        line.set_3d_properties(np.tile(np.arange(frame), len(quark_masses)))
        return line,

    ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True)
    ani.save(f'big_bang_simulation_3d_{tunneling_probability}.gif', writer='pillow')
    plt.show()

    # --- Plotly Mass Evolution (3D) ---
    X, Y = np.meshgrid(particle_speeds[0], np.arange(num_steps))  # Create 2D meshgrid for x and y
    fig = go.Figure(data=[go.Surface(z=particle_masses_evolution[j], x=X, y=Y, colorscale='Viridis', name=quark.capitalize()) for j, quark in enumerate(quark_masses.keys())])
    fig.update_layout(title=f"Big Bang Simulation: Mass Evolution (Tunneling Probability: {tunneling_probability})", autosize=False, width=800, height=600, margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

    # --- Plotly Tunneling Effect (3D) ---
    X, Y = np.meshgrid(particle_speeds[0], np.arange(num_steps))  # Create 2D meshgrid for x and y
    fig = go.Figure(data=[go.Surface(z=tunneling_steps[j], x=X, y=Y, colorscale='Blues', name=quark.capitalize()) for j, quark in enumerate(quark_masses.keys())])
    fig.update_layout(title=f"Big Bang Simulation: Tunneling Effect (Tunneling Probability: {tunneling_probability})", autosize=False, width=800, height=600, margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

    # --- Correlation Analysis ---
    df = pd.DataFrame({
        'Speed': particle_speeds.flatten(),
        'Temperature': particle_temperatures.flatten(),
        'Mass': particle_masses_evolution.flatten(),
        'Tunneling': tunneling_steps.flatten()
    })

    correlation_matrix = df.corr()
    correlation_matrices.append(correlation_matrix)

    print("Correlation Matrix:")
    print(correlation_matrix)

    # Print calculated masses at the end of the simulation
    print(f"Calculated masses at the end of the simulation (Tunneling Probability: {tunneling_probability}):")
    for j, quark in enumerate(quark_masses.keys()):
        print(f"{quark}: {particle_masses_evolution[j, -1] / GeV_to_J:.4e} GeV")
    print(f"Real masses:")
    for quark, mass in quark_masses.items():
        print(f"{quark}: {mass:.4e} GeV")

# Create a table of correlation matrices
correlation_matrices_df = pd.DataFrame(correlation_matrices)
print("Correlation Matrices for Different Tunneling Probabilities:")
print(correlation_matrices_df)
