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

def plot_uniform_distribution(samples):
    # Generate and plot random numbers from a uniform distribution
    uniform_random_numbers = np.random.uniform(0, 1, samples)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=uniform_random_numbers, histnorm='probability'))
    fig.update_layout(title="Uniform Distribution", xaxis_title="Value", yaxis_title="Probability")
    plot(fig, filename="uniform_distribution")

#plot_uniform_distribution(1000)

def plot_uniform_random_field(num_particles, space_size):
    x = np.random.uniform(-space_size/2, space_size/2, num_particles)
    y = np.random.uniform(-space_size/2, space_size/2, num_particles)
    z = np.random.uniform(-space_size/2, space_size/2, num_particles)

    # Plotting the particle field
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
    fig.update_layout(
        title="Uniform Random Particle Field in 3D",
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis')
        )
    )
    plot(fig, "3d_uniform_random_field")

#plot_uniform_random_field(500, 10)

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
'''
# Generate 1000 random numbers from a exponential distribution
for material in materials:
    print(f"Material: {material.name}")
    samples = 1000
    macroscopic_cross_section= material.absorption_cross_section
    uniform_random_numbers = np.random.uniform(0, 1, samples)

    # Generate the exponential distribution
    exponential_random_numbers = - np.log(uniform_random_numbers) / macroscopic_cross_section

    max_distance = max(exponential_random_numbers)
    # Bin the data
    counts, bin_edges = np.histogram(exponential_random_numbers, bins=100)

    # Compute bin centres (for plotting and fitting, so we don't calculate the bins twice)
    # Bin edges are the boundaries of the bins, so we need to average them to get the centres
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    # Compute the survival function: cumulative sum from right to left
    cumulative_counts = np.cumsum(counts[::-1])[::-1]

    # Create histogram trace using binned data
    histogram_trace = go.Scatter(
        x=bin_centres,
        y=cumulative_counts,
        mode='markers',
        name=f"Simulated survival function, {material.name}"
    )

    theoretical_x = np.linspace(0, max_distance, samples)
    theoretical_y = samples * np.exp(-theoretical_x * macroscopic_cross_section)

    theoretical_trace = go.Scatter(
        x=theoretical_x,
        y=theoretical_y,
        mode='lines',
        name=f"Theoretical survival function, {material.name}"
    )

    line_trace = go.Scatter(
        x=[0, max_distance],
        y=[samples / np.e] * 2,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name=f"{material.name} - N_0/e Line"
    )

    # Combine all traces and plot
    layout = go.Layout(
        title=f"Survival function for {material.name}",
        xaxis=dict(title="Distance (cm)"),
        yaxis=dict(title="Counts"),
        legend=dict(x=0, y=1)
    )

    fig = go.Figure(
        data=[histogram_trace, theoretical_trace, line_trace], layout=layout)
    plot(fig, f"exponential_distribution_{material.name}")

    x2 = bin_centres[cumulative_counts > 0]
    y2 = np.log(cumulative_counts[cumulative_counts > 0])

    weights = np.sqrt(cumulative_counts[cumulative_counts > 0])

    fit, cov = np.polyfit(x2, y2, 1, cov=True)

    slope = fit[0]
    slope_uncertainty = np.sqrt(cov[0][0])
    simulated_attenuation_length = -1 / slope
    uncertainty = slope_uncertainty / slope**2

    print(f"Simulated attenuation length, using binned, for {material.name}: {simulated_attenuation_length:.2f} ± {uncertainty:.2f}")
    print(f"Theoretical attenuation length for {material.name}: {1/macroscopic_cross_section:.2f}")

    """
    In accordance with the assignment instructions, the first figure displays a binned histogram of neutron distances
    with an overlaid theoretical exponential curve. While broadly consistent, the simulated data shows a slight rightward
    bias due to binning: counts are grouped at bin midpoints, whereas the underlying exponential decay is continuous.

    To address this, a second plot uses ranked raw data and plots the natural logarithm of the number of neutrons that
    reached at least a given distance. This approach eliminates binning error and results in a significantly closer match
    to the theoretical decay curve, as shown by the alignment between the simulated and expected values. The improved
    accuracy is also reflected in the fitted attenuation length, which closely matches the theoretical prediction.
    """

    sorted_data = np.sort(exponential_random_numbers)
    ranks = np.arange(len(sorted_data), 0, -1)  # from N to 1

    fit, cov = np.polyfit(sorted_data, np.log(ranks), 1, cov=True)

    slope = fit[0]
    slope_uncertainty = np.sqrt(cov[0][0])
    simulated_attenuation_length = -1 / slope
    uncertainty = slope_uncertainty / slope**2

    print(f"Simulated attenuation length, using ranked, for {material.name}: {simulated_attenuation_length:.2f} ± {uncertainty:.2f}")
    print(f"Theoretical attenuation length for {material.name}: {1/macroscopic_cross_section:.2f}")
'''
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

#plot_random_unit_vectors(1000)

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
'''
for material in materials:
    macroscopic_cross_section= material.absorption_cross_section
    
    positions_iterator = random_walk(macroscopic_cross_section)
    positions = [next(positions_iterator) for _ in range(50)]

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
    plot(fig, f"random_walk_{material.name}")

    # Do some random walks and plot the results
'''
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

def random_walks(slabs, counts):
    return [ random_walk(slabs) for _ in range(counts)]

def count_fates(slabs, counts):
    return Counter([walk[0] for walk in random_walks(slabs, counts)])

def plot_random_walks(slabs, counts):
    walks = random_walks(slabs, counts)

    layout = go.Layout(
        title=f"Random Walk in {'_'.join([slab.material.name+str(slab.width) for slab in slabs])}",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )
    show_legend = {
        "Transmitted": True,
        "Reflected": True,
        "Absorbed": True
    }
    colours = {
        "Transmitted": "blue",
        "Reflected": "red",
        "Absorbed": "green"
    }
    fate_counts = Counter([walk[0] for walk in walks])

    def walk_to_scatter(walk):
        x, y, z = zip(*walk[1])
        showlegend = show_legend[walk[0]]
        show_legend[walk[0]] = False #only show the legend for the first walk of each type
        return go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=2), name=f'{walk[0]} ({fate_counts[walk[0]]})', line=dict(color=colours[walk[0]]), showlegend=showlegend)

    fig = go.Figure(data=[walk_to_scatter(walk) for walk in walks], layout=layout)
    plot(fig, f"rw_{'_'.join([slab.material.name+str(slab.width) for slab in slabs])}_{counts}")

#plot_random_walks([Slab(water, 0.000001)], 100)
#plot_random_walks([Slab(water, 1)], 100)
#plot_random_walks([Slab(water, 3)], 100)
#plot_random_walks([Slab(water, 6)], 100)

def binomial_standard_error(probability, trials):
    return np.sqrt(probability * (1 - probability) / trials)

def summarise_trial(material, trials, width):
    fates = count_fates([Slab(material, width)], trials)
    print(f"Material: {material.name}, Width: {width} cm, Trials: {trials}")
    for fate, count in fates.items():
        probability = count / trials
        error = binomial_standard_error(probability, trials)
        print(f"  {fate}: {count} ({probability:.2f} ± {error:.2f})")
    print("==========================")

#summarise_trial(water, 100, 1)

def plot_width_vs_fates(trials, steps):
    materials = [(water, 10), (lead, 50), (graphite, 50)]
    for material, max_width in materials:
        widths = np.linspace(0.000001, max_width, steps)
        data = { }
        for width in widths:
            fates = count_fates([Slab(material, width)], trials)
            for fate, count in fates.items():
                probability = count / trials
                error = trials * binomial_standard_error(probability, trials)
                if fate not in data:
                    data[fate] = []
                
                data[fate].append((width, count, error))
       
        fig = go.Figure()

        def exponential_func(x, attenuation_length):
             return trials * np.exp(-x / attenuation_length)
        widths, counts, errors = zip(*data["Transmitted"])

        popt, pcov = curve_fit(exponential_func, widths, counts)
        attenuation_length_curve_fit = popt[0]
        attenuation_length_error_curve_fit = np.sqrt(pcov[0][0])
        print(f"Attenuation length for {material.name}: {attenuation_length_curve_fit:.2f} ± {attenuation_length_error_curve_fit:.2f}")

        theoretical_x = np.linspace(0, max_width, 1000)
        theoretical_y_curve_fit = exponential_func(theoretical_x, attenuation_length_curve_fit)
        theoretical_trace_curve_fit = go.Scatter(
            x=theoretical_x,
            y=theoretical_y_curve_fit,
            mode='lines',
            name=f"Fitted survival function, {material.name}"
        )
        
        fig.add_trace(theoretical_trace_curve_fit)

        for fate_name in data:
            widths, counts, errors = zip(*data[fate_name])
            fig.add_trace(go.Scatter(
                x=widths,
                y=counts,
                mode='markers',
                name=f"{material.name} {fate_name}",
                error_y=dict(
                    type='data',
                    array=errors,
                    visible=True
                )
            ))
        fig.update_layout(
            title=f"Random Walks for {material.name}",
            xaxis_title="Width (cm)",
            yaxis_title="Probability",
            legend=dict(x=0, y=1)
        )
        plot(fig, f"width_vs_fates_{material.name}")

#plot_width_vs_fates(1000, 100)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")

check_results = count_fates([Slab(graphite, 10), Slab(graphite, 10)], 1000)
print(check_results)

check_results = count_fates([Slab(graphite, 10)], 1000)
print(check_results)

check_results = count_fates([Slab(graphite, 10)], 1000)
print(check_results)