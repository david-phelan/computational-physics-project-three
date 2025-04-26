import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from collections import Counter
import time
import os
import shutil

#delete out directory if it exists
if os.path.exists("out"):
    shutil.rmtree("out")
os.makedirs("out/html", exist_ok=True)
os.makedirs("out/png", exist_ok=True)

start_time = time.time()

def plot(fig, filename):
    fig.write_html(f"out/html/{filename}.html")
    # 4k HD resolution
    # fig.write_image(f"out/png/{filename}.png", width=3840, height=2160)
    fig.write_image(f"out/png/{filename}.png", width=1920, height=1080)
    # fig.write_image(f"out/png/{filename}.png", width=800, height=600)

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
    
class Slab:
    def __init__(self, material, width):
        self.material = material
        self.width = width

# Define materials with their properties
water = Material.construct_from_microscopic_properties(
    "Water", 0.6652, 103.0, 1.0, 18.0153)

lead = Material.construct_from_microscopic_properties(
    "Lead", 0.158, 11.221, 11.35, 207.2)

graphite = Material.construct_from_microscopic_properties(
    "Graphite", 0.0045, 4.74, 1.67, 12.011)

materials = [water, lead, graphite]

# Week 2

def random_unit_vectors(num_vectors):
    """
    Generate random unit vectors distributed over a sphere.

    You'd think this works, but notice changes in azimuthal angle
    correspond to a larger shift in distance for the points when
    the polar angle is 90 degrees, vs 0 degress (when it corresponds to
    literally no distance difference!) so you end up with a
    clusering at the poles.
    """
    theta = np.random.uniform(0, 2 * np.pi, num_vectors)  # azimuthal angle
    phi = np.random.uniform(0, np.pi, num_vectors)  # polar angle

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return x, y, z

def random_uniform_unit_vectors(num_vectors):
    """
    Generate random unit vectors uniformly distributed over a sphere.
    """

    z = np.random.uniform(-1, 1, num_vectors)  # equivalent to cos(theta)
    phi = np.random.uniform(0, 2 * np.pi, num_vectors)

    x = np.sqrt(1 - z**2) * np.cos(phi)
    y = np.sqrt(1 - z**2) * np.sin(phi)
    # z is already correct

    return x, y, z

def plot_random_unit_vectors(num_vectors):
    layout = go.Layout(
        title="Random Unit Vectors on a Sphere",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )
    x, y, z = random_uniform_unit_vectors(num_vectors)
    scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2))
    fig = go.Figure(data=[scatter], layout=layout)
    plot(fig, "random_unit_vectors")


def random_uniform_unit_vector():
    z = np.random.uniform(-1, 1)
    phi = np.random.uniform(0, 2 * np.pi)

    x = np.sqrt(1 - z**2) * np.cos(phi)
    y = np.sqrt(1 - z**2) * np.sin(phi)

    return x, y, z

def random_uniform_path_length(cross_section):
    uniform_random_numbers = np.random.uniform(0, 1)
    return - np.log(uniform_random_numbers) / cross_section

def random_walk(cross_section):
    x, y, z = (0, 0, 0)
    while True:
        step_size = random_uniform_path_length(cross_section)
        dx, dy, dz = random_uniform_unit_vector()
        x += step_size * dx
        y += step_size * dy
        z += step_size * dz
        yield (x, y, z)

def random_walk(slabs):
    position = (0, 0, 0)
    direction = (1, 0, 0)
    path = [position]
    max_cross_section = max(slab.material.total_cross_section for slab in slabs)
    max_width = sum(slab.width for slab in slabs)

    def get_slab_map():
        x_start = 0
        for slab in slabs:
            x_end = x_start + slab.width
            yield (x_start, x_end, slab.material, slab.material.total_cross_section / max_cross_section)
            x_start = x_end

    def get_material_at(x):
        return next(((m, p) for start, end, m, p in slab_positions if start <= x < end), None)

    slab_positions = list(get_slab_map())

    while True:
        step_size = random_uniform_path_length(max_cross_section)
        x = position[0] + step_size * direction[0]
        y = position[1] + step_size * direction[1]
        z = position[2] + step_size * direction[2]
        position = (x, y, z)

        if x < 0:
            path.append(position)
            return "Reflected", path
        if x > max_width:
            path.append(position)
            return "Transmitted", path

        material, p_real = get_material_at(x)
        if material is None:
            raise ValueError("Position out of bounds")

        if np.random.uniform(0, 1) > p_real:
            continue
        else:
            if np.random.uniform(0, 1) < material.absorption_probability:
                path.append(position)
                return "Absorbed", path
            else:
                direction = random_uniform_unit_vector()
                path.append(position)

random_walk([Slab(water, 1.0)])