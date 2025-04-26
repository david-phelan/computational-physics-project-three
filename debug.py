import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo

class Material:
    BARN_TO_CM2 = 1e-24  # cm²
    AVOGADRO_CONSTANT = 6.02214076e23  # 1/mol

    def __init__(self, name, scattering_cross_section, absorption_cross_section):
        self.name = name
        self.total_cross_section = scattering_cross_section + absorption_cross_section
        self.scattering_cross_section = scattering_cross_section
        self.absorption_cross_section = absorption_cross_section
        self.absorption_probability = absorption_cross_section / self.total_cross_section

    @staticmethod
    def construct_from_microscopic_properties(name, sigma_a, sigma_s, density, molar_mass):
        number_density = Material.AVOGADRO_CONSTANT * density / molar_mass

        return Material(
            name,
            number_density * sigma_s * Material.BARN_TO_CM2,
            number_density * sigma_a * Material.BARN_TO_CM2
        )

# Define materials with their properties
water = Material.construct_from_microscopic_properties(
    "Water", 0.6652, 103.0, 1.0, 18.0153)

lead = Material.construct_from_microscopic_properties(
    "Lead", 0.158, 11.221, 11.35, 207.2)

graphite = Material.construct_from_microscopic_properties(
    "Graphite", 0.0045, 4.74, 1.67, 12.011)

materials = [water, lead, graphite]

def random_uniform_path_length(cross_section):
    return -np.log(np.random.uniform(0, 1)) / cross_section

def generate_random_unit_vector():

    theta = np.arccos(np.random.uniform(-1, 1))
    phi = np.random.uniform(0, 2*np.pi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    print
    return x, y, z

def random_walk_slab(cross_section, slab_thickness, absorbtion_probability):
    x, y, z = (0, 0, 0)
    yield (x, y, z)
    dx, dy, dz = (1, 0, 0)  # Initial direction
    while True:
        step_size = random_uniform_path_length(cross_section)
        x += step_size * dx
        y += step_size * dy
        z += step_size * dz
        if np.random.uniform(0, 1) < absorbtion_probability:
            # Absorption occurs
            break
        elif x > slab_thickness:
            break  # Exit the slab
        elif x < 0:
            # Reflect at the boundary
            break
        else:
            # Scattering occurs
            dx, dy, dz = generate_random_unit_vector()
        print(x,y,z)
        yield (x, y, z)

print("Absorbtion probability:")
for material in materials:
    print(f"{material.name}: {material.absorption_probability:.5f}")

for material in materials:
    macroscopic_cross_section= material.absorption_cross_section
    print(f"Material: {material.name}, Macroscopic Cross Section: {macroscopic_cross_section:.5f} cm^-1")

    positions_iterator = random_walk_slab(macroscopic_cross_section, 10, material.absorption_probability)
    positions = list(positions_iterator)
    print(positions)

    x, y, z = zip(*positions)
    scatter = go.Scatter3d(x=x, y=y, z=z, mode='lines', marker=dict(size=2))
    layout = go.Layout(
        title=f"Random Walk in {material.name}",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

    # Do some random walks and plot the results