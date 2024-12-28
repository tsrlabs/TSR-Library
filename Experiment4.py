import numpy as np
import plotly.graph_objects as go
import os
import json
import imageio
import plotly.io as pio

# Constants and parameters
results_dir = "quantum_convergence_results"
num_points = 100  # Adjust based on your simulation data structure
L = 1e-8  # Length of the space (m)
x = np.linspace(-L, L, num_points)

# Function to load data
def load_data(run):
    filename = os.path.join(results_dir, f'combined_simulation_results_run_{run}.json')
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# Load results and print shapes for debugging
data_samples = [load_data(run) for run in range(10)]  # Adjust the range based on your available data
for i, sample in enumerate(data_samples):
    print(f"Sample {i} shape: {np.array(sample['correlation_combined']).shape}")

# Verify the shape is as expected
psi_t_samples = []
for sample in data_samples:
    psi = np.array(sample['correlation_combined'])
    if psi.shape == (num_points,):  # Adjust if needed
        psi_t_samples.append(psi.reshape((10, 10)))  # Assuming 10x10 if you know exact reshaping

psi_t = np.mean(psi_t_samples, axis=0)

# Create 3D plot frames
frames = [go.Frame(data=[go.Surface(z=psi_t, x=x, y=np.linspace(0, 1e-15, 10), colorscale='Viridis')], 
                   name=f'frame_{t}') for t in range(10)]

fig = go.Figure(
    data=[go.Surface(z=psi_t, x=x, y=np.linspace(0, 1e-15, 10), colorscale='Viridis')],
    layout=go.Layout(
        title="Particle Tunneling Over Time with TSR",
        scene=dict(
            xaxis_title='Position (m)',
            yaxis_title='Time (s)',
            zaxis_title='Probability Density'
        ),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]
            }]
        }]
    ),
    frames=frames
)

# Save the animation as an HTML file
pio.write_html(fig, file='particle_tunneling_animation.html', auto_open=True)

# Convert HTML to images
pio.write_image(fig, 'temp_plot.png')  # Save each frame as an image

# Create video
with imageio.get_writer('particle_tunneling_animation.mp4', fps=15) as writer:
    for i in range(10):
        writer.append_data(imageio.imread(f'temp_plot_frame_{i}.png'))
