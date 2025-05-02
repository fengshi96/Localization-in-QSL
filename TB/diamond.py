import math 
import sys, os
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import numpy.typing as npt
import scipy
import primme
from collections import Counter
from koala.plotting import plot_edges, plot_plaquettes,plot_vertex_indices
from koala.graph_color import color_lattice
from koala.flux_finder import fluxes_from_ujk, fluxes_to_labels, ujk_from_fluxes
import koala.hamiltonian as ham
from koala.lattice import Lattice
from koala import chern_number as cn
from koala.example_graphs import honeycomb_lattice, n_ladder
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from scipy import sparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.sparse.linalg import spsolve
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from PIL import Image
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



def sierpinskicoor(n):
    """
    Recursively generate the coordinates of the Sierpinski gasket at level n.

    Parameters:
    n (int): The level of the Sierpinski gasket.

    Returns:
    np.ndarray: An array of shape (3^n, 2) containing the coordinates.
    """
    o = np.array([0.1, 0.1])
    delta1 = np.array([1.0, 0.0]) 
    delta2 = np.array([0.5, math.sqrt(3) / 2])
    if n == 1:
        # Base case: Return the initial triangle vertices
        return np.array([o, o + delta1, o + delta2])
    else:
        # Get coordinates from the previous level
        snm1 = sierpinskicoor(n - 1)
        shift_factor = 2 ** (n - 1)
        s1 = shift_factor * delta1
        s2 = shift_factor * delta2

        # Shift the previous coordinates to create two new triangles
        snm1_shifted_s1 = snm1 + s1
        snm1_shifted_s2 = snm1 + s2

        # Combine all coordinates
        coordinates = np.vstack((snm1, snm1_shifted_s1, snm1_shifted_s2))
        return coordinates

def gen_bonds(n):
    if n == 1:
        bonds_array = np.array([[0, 1], [1, 2], [2, 0]])
        return bonds_array
    else:
        bnm1 = gen_bonds(n - 1)
        Ns_prev = 3 ** (n - 1)
        
        # Shift bonds
        bnm1_shifted1 = bnm1 + Ns_prev
        bnm1_shifted2 = bnm1 + 2 * Ns_prev
        
        # Combine bonds
        bonds_n = np.vstack((bnm1, bnm1_shifted1, bnm1_shifted2))
        
        # Calculate sum_prev
        if n >= 3:
            sum_prev = sum(3 ** i for i in range(1, n - 1))
        else:
            sum_prev = 0
        
        # Additional bonds (adjusted for zero-based indexing)
        bond_a = [sum_prev + 2 - 1, Ns_prev + 1 - 1]     # Bond A
        bond_b = [Ns_prev - 1, 2 * Ns_prev + 1 - 1]      # Bond B
        bond_c = [2 * Ns_prev - 1, 5 * sum_prev + 8 - 1] # Bond C
        
        # Add the additional bonds
        additional_bonds = np.array([bond_a, bond_b, bond_c])
        bonds_n = np.vstack((bonds_n, additional_bonds))
        
        return bonds_n
    


def diamond_chain_vertices(n):
    """
    generate the coordinates of the diamond chain of n 4-site unit cells 

    Parameters:
    n (int): number of unit celss

    Returns:
    np.ndarray: An array containing the coordinates.
    """
    oshiftx = 0.1
    oshifty = 2
    ydist = 0.5
    o = np.array([oshiftx, oshifty])
    j1 = np.array([0.5 + oshiftx, ydist + oshifty])
    j2 = np.array([0.5 + oshiftx, -ydist + oshifty])
    j3 = np.array([1.0 + oshiftx, 0.0 + oshifty]) 

    vertices = []
    for i in range(n):
        shift = np.array([1.6 * i, 0.0])
        vertices.append(o + shift)
        vertices.append(j1 + shift)
        vertices.append(j2 + shift)
        vertices.append(j3 + shift)

    # vertices.append([1.6 * n + oshiftx, oshifty])
    # vertices.append([1.6 * n // 2 + oshiftx, -2.5])

    vertices = np.array(vertices)
    print(vertices)
    return vertices

        
def diamond_chain_bonds(n):
    """
    generate the bonds of the diamond chain of n 4-site unit cells 
    Parameters:
    n (int): number of unit celss

    Returns:
    np.ndarray: An array containing the coordinates.
    """
    bonds = []
    for i in range(n):
        shift = 4 * i
        bonds.append([shift + 0, shift + 1])
        bonds.append([shift + 0, shift + 2])
        bonds.append([shift + 1, shift + 3])
        bonds.append([shift + 2, shift + 3])
        bonds.append([shift + 1, shift + 2])
        if i != n - 1:
            bonds.append([shift + 3, shift + 4])
    
    # bonds.append([0, 15])
    bonds = np.array(bonds)
    print(bonds)
    return bonds


def diamond_chain_lattice(n):
    """
    generate the diamond chain lattice of n 4-site unit cells 
    Parameters:
    n (int): number of unit celss

    Returns:
    np.ndarray: An array containing the coordinates.
    """
    vertices = diamond_chain_vertices(n)
    bonds = diamond_chain_bonds(n)
    new_vet_positions = vertices / (np.max(vertices)*1.05) + np.array([0.01, 0.5])
    
    # Create a Lattice object
    modified_lattice = Lattice(new_vet_positions, bonds, np.zeros_like(bonds))
    coloring_solution = color_lattice(modified_lattice)
    return modified_lattice, coloring_solution



def regular_Sierpinski(fractal_level=1, remove_corner=False):
    """
    Generate Sierpinski at the specified fractal level
    level >= 1; 1 gives a triangle
    """
    modified_lattice = sierpinskicoor(fractal_level)
    print(modified_lattice)
    if fractal_level == 0:
       raise ValueError("fractal level must be >= 1")

    new_vet_positions = modified_lattice.copy()
    new_edge_indices = gen_bonds(fractal_level)
    print(new_edge_indices)
    # print(np.max(new_vet_positions))

    new_vet_positions = new_vet_positions / (np.max(new_vet_positions)*1.05) + np.array([0.02, 0.02])

    if remove_corner:
        # let us first select the two-coordinated vertices 
        flattened_vertices = new_edge_indices.flatten()
        vertex_counts = Counter(flattened_vertices)
        two_coord_vertcies = [vertex for vertex, count in vertex_counts.items() if count == 2]
        # there should be 3 of them i.e. the 3 corners of the sierpinski triangle
        assert(len(two_coord_vertcies) == 3)

        # an ancilla qubit to be conneced to the 2-coordinated corners
        ancilla = [0.5, 0.97]
        new_vet_positions = np.vstack([new_vet_positions, ancilla])
        ancilla_indx = len(new_vet_positions) - 1
        ancilla_edge_x = [ancilla_indx, two_coord_vertcies[0]]
        ancilla_edge_y = [ancilla_indx, two_coord_vertcies[1]]
        ancilla_edge_z = [ancilla_indx, two_coord_vertcies[2]]
        new_edge_indices = np.vstack([new_edge_indices, ancilla_edge_x, ancilla_edge_y, ancilla_edge_z])
        # print(new_edge_indices)

    new_edge_crossing = np.zeros_like(new_edge_indices)
    modified_lattice = Lattice(new_vet_positions, new_edge_indices, new_edge_crossing)
    print("Number of vertices =", modified_lattice.n_vertices)

    coloring_solution = color_lattice(modified_lattice)

    return modified_lattice, coloring_solution



def flux_sampler(modified_lattice, num_fluxes, seed=None):
    if seed is not None:
        np.random.seed(seed)  

    num_plaquettes = len(modified_lattice.plaquettes)
    num_fluxes = num_fluxes  # Replace with the desired number of +1 fluxes

    # Generate a base array of -1 (no flux)
    target_flux = np.full(num_plaquettes, -1, dtype=np.int8)

    indices_with_flux = np.random.choice(
        num_plaquettes, num_fluxes, replace=False
    )

    target_flux[indices_with_flux] = 1

    return target_flux


def diag_maj(modified_lattice, coloring_solution, ujk, method='dense', k=1, max_ipr_level=5):
# constructing and solving majorana Hamiltonian for the gs sector
    if coloring_solution is not None:
        maj_ham = ham.majorana_hamiltonian(modified_lattice, coloring_solution, ujk)
    else:
        maj_ham = ham.majorana_hamiltonian(modified_lattice, None, ujk)
    
    if method == 'dense':
        maj_energies, eigenvectors = scipy.linalg.eigh(maj_ham)
    else:
        smaj_ham = sparse.csr_matrix(maj_ham)
        maj_energies, eigenvectors = primme.eigsh(smaj_ham, k=k, tol=1e-10, which=0)

    gap = min(np.abs(maj_energies))
    ipr_values = np.array([(np.sum(np.abs(eigenvectors)**(2*q), axis=0)) for q in np.arange(2, max_ipr_level+1, 1)])
    print('Gap =', gap)

    epsilon = 1e-8
    num_zero_energy_levels = np.sum(np.abs(maj_energies) < epsilon)
    print(f"Zero-energy levels: {num_zero_energy_levels}")

    data = {}
    data['gap'] = gap
    data['ipr'] = ipr_values
    data['ujk'] = ujk
    data['energies'] = maj_energies
    data['eigenvectors'] = eigenvectors


    return data



def complex_fluxes_to_labels(fluxes: np.ndarray) -> np.ndarray:
    """Remaps fluxes from the set {1,-1, +i, -i} to labels in the form {0,1,2,3} for plotting.

    Args:
        fluxes (np.ndarray): Fluxes in the format +1 or -1 or +i or -i

    Returns:
        np.ndarray: labels in [0(+1),1(-1),2(+i),3(-i)] to which I later assign the color_scheme=np.array(['w','lightgrey','wheat', 'thistle'])
    """
    flux_labels = np.zeros(len(fluxes), dtype=int)
    for i, p in enumerate(fluxes):
        if p == 1:
            flux_labels[i] = 0
        elif p == -1:
            flux_labels[i] = 1
        elif p == 1.j:
            flux_labels[i] = 2
        elif p == -1.j:
            flux_labels[i] = 3
    
    return flux_labels

def plot_dist_smooth(lattice, wave_function_distribution, resolution=300, cmap='gist_yarg', vmin=0, vmax=1, 
                     label=r"$|\psi_i|^2$", filename="real_space_wf.pdf", s=100, show_lattice=False, interpolate=False):
    """
    Plot the wave function |ψ_i|^2 distribution as a smooth field on the lattice.

    Args:
        lattice (Lattice): The lattice object.
        wave_function_distribution (np.ndarray): Probability distribution e.g. |ψ_i|^2.
        resolution (int): Resolution of the interpolation grid.
    """
    # Extract vertex positions
    positions = lattice.vertices.positions
    x, y = positions[:, 0], positions[:, 1]

    # Normalize the wave function distribution
    normalized_distribution = wave_function_distribution / np.max(wave_function_distribution)
    if interpolate:
        # Create a grid for interpolation
        xi = np.linspace(np.min(x), np.max(x), resolution)
        yi = np.linspace(np.min(y), np.max(y), resolution)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate the wave function onto the grid
        zi = griddata((x, y), normalized_distribution, (xi, yi), method='cubic')

        # Replace NaN values introduced during interpolation with zeros
        zi = np.nan_to_num(zi)

        # Plot the smooth field
        fig, ax = plt.subplots(1,1, figsize=(9, 8))
        plot_edges(lattice, ax= ax, color='black', alpha=0.2, lw=0.5)
        scatter = plt.imshow(
            zi, extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
            origin='lower', cmap=cmap, alpha=0.9, vmin=vmin, vmax=vmax
        )
    else:
        # Sort the points by normalized_distribution to ensure higher values are plotted last
        sorted_indices = np.argsort(normalized_distribution)
        x = x[sorted_indices]
        y = y[sorted_indices]
        normalized_distribution = normalized_distribution[sorted_indices]

        plt.figure(figsize=(9, 8))

        scatter = plt.scatter(
            x, y,
            c=normalized_distribution,  # Color by normalized |ψ_i|^2
            cmap=cmap,              # Smooth colormap
            s=s,                      # Size of the scatter points
            alpha=0.7,
            vmax=vmax,                  # Transparency
            vmin=vmin,
            edgecolors='none'           # Remove edge colors for smoothness
        )

    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=18)
    cbar.ax.tick_params(labelsize=18)  

    # Clean up the frame and ticks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    if show_lattice:
        plot_edges(lattice, color='black', lw=0.5, alpha=0.2)

    plt.savefig(filename, dpi=300,bbox_inches='tight')



def perturb_ground_state(ground_state, site_index, perturbation_strength=0.1):
    """
    Perturb the ground state wavefunction by modifying its amplitude locally.

    Args:
        ground_state (np.ndarray): The ground state wavefunction.
        site_index (int): Index of the site to perturb.
        perturbation_strength (float): Strength of the perturbation.

    Returns:
        np.ndarray: Perturbed wavefunction.
    """
    perturbed_state = ground_state.copy()
    perturbed_state[site_index] += perturbation_strength
    return perturbed_state / np.linalg.norm(perturbed_state)  # Normalize


def time_evolution(energies, eigenvectors, psi_0, times):
    """
    Perform time evolution of a wavefunction.

    Args:
        energies (np.ndarray): Eigenvalues of the Hamiltonian.
        eigenvectors (np.ndarray): Eigenvectors of the Hamiltonian.
        psi_0 (np.ndarray): Initial wavefunction.
        times (np.ndarray): Array of time points.

    Returns:
        np.ndarray: Time-evolved wavefunctions.
    """
    # Expand initial wavefunction in eigenbasis
    coefficients = np.dot(eigenvectors.T.conj(), psi_0)

    # Precompute exponential time evolution factors
    evolution_factors = np.exp(-1j * np.outer(energies, times))

    # Compute time-evolved wavefunction
    psi_t = np.dot(eigenvectors, coefficients[:, np.newaxis] * evolution_factors)
    return psi_t





def get_closest_site_to_center(lattice):
    """
    Get the site index closest to the center of the lattice canvas.

    Args:
        lattice (Lattice): The lattice object containing vertex positions.

    Returns:
        int: The index of the site closest to the center.
    """
    # Extract vertex positions
    positions = lattice.vertices.positions
    x, y = positions[:, 0], positions[:, 1]

    # Compute the center of the canvas
    x_center, y_center = (np.min(x) + np.max(x)) / 2, (np.min(y) + np.max(y)) / 2

    # Compute Euclidean distances from the center
    distances = np.sqrt((x - x_center)**2 + (y - y_center)**2)

    # Get the index of the closest site
    closest_site_index = np.argmin(distances)

    return closest_site_index


def create_time_evolution_animation(lattice, wavefunctions, total_time, output_gif, target_flux=None, cmap="Purples", fps=5):
    """
    Create a simple animation of wavefunction evolution.

    Args:
        lattice: Lattice structure containing vertex positions.
        wavefunctions: A 2D array (time steps x lattice sites) of wavefunction amplitudes.
        output_gif: The name of the output GIF file.
        cmap: Colormap for visualizing wavefunction amplitudes.
        fps: Frames per second for the animation.
    """
    positions = lattice.vertices.positions  # Get lattice site positions
    x, y = positions[:, 0], positions[:, 1]  # Split into x and y coordinates
    wavefunctions = np.abs(wavefunctions)**2

    total_frames = wavefunctions.shape[1]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(x, y, c=np.zeros_like(x), cmap=cmap, s=100, vmin=0, vmax=0.05)
    plot_edges(lattice, color='black', lw=0.5, alpha=0.2)
    if target_flux is not None:
        plot_plaquettes(lattice, ax=ax, labels = fluxes_to_labels(target_flux), color_scheme=np.array(['lightgrey','w','deepskyblue', 'wheat']))
    ax.axis("equal")
    ax.axis("off")

    time_text = ax.text(
        0.02, 0.98, "t=0.00", transform=ax.transAxes, fontsize=12, verticalalignment="top"
    )

    # Update function for the animation
    def update(frame):
        # Get current wavefunction and sort it by amplitude
        wf = wavefunctions[frame]
        sorted_indices = np.argsort(wf)  # Sort in ascending order
        sorted_x = x[sorted_indices]
        sorted_y = y[sorted_indices]
        sorted_wf = wf[sorted_indices]

        # ax.clear()
        ax.axis("equal")
        ax.axis("off")
        
        scatter = ax.scatter(
            sorted_x, sorted_y,
            c=sorted_wf,
            cmap=cmap,
            s=100,
            vmin=0,
            vmax=0.1
        )
        print('updating frame = ', frame)
        t = frame * (total_time / total_frames)
        print(np.round(t, 2))
        time_text.set_text(f"t={np.round(t, 2)}")
        return scatter

    plt.tight_layout(pad=0)
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(wavefunctions), interval=1000 / fps)

    # Save the animation
    ani.save(output_gif, writer="imagemagick", fps=fps, dpi=150)
    print(f"Animation saved to {output_gif}")


def construct_ujk(n, p):
    """
    Construct ujk list according to the rule:
    Initially: ujk = [1, -1, -1, -1, 1, -1]
    For each subsequent iteration:
        ujk += [1, -1, -1, -1, 1, -1]
        With probability p, flip the fourth '+1' (0-based indexing) to '-1'

    Parameters:
    n (int): Number of unit cells
    p (float): Probability to flip the fourth '+1'

    Returns:
    list: resulting ujk list
    """
    ujk = [1, -1, -1, -1, 1, -1]

    for _ in range(n - 1):
        next_segment = [1, -1, -1, -1, 1, -1]
        if np.random.rand() < p:
            # flip the fourth '+1', which is at index 4 (0-based)
            next_segment[4] = -1
            next_segment[5] = 1
        ujk += next_segment

    return np.array(ujk)



def construct_ujk_random(n, p):
    """
    Construct ujk list according to the rule:
    Initially: ujk = [1, -1, -1, -1, 1, -1]
    For each subsequent iteration:
        ujk += [1, -1, -1, -1, 1, -1]
        With probability p, flip the fourth '+1' (0-based indexing) to '-1'

    Parameters:
    n (int): Number of unit cells
    p (float): Probability to flip the fourth '+1'

    Returns:
    list: resulting ujk list
    """
    ujk = [1, -1, -1, -1, 1, -1]

    for _ in range(n - 1):
        next_segment = [1, -1, -1, -1, 1, -1]
        if np.random.rand() < p:
            # flip the fourth '+1', which is at index 4 (0-based)
            next_segment[2] = 1
        ujk += next_segment

    return np.array(ujk)



def construct_ujk_antialigned(n, p):
    """
    Construct ujk list according to the rule:
    Initially: ujk = [1, -1, -1, -1, 1, -1]
    For each subsequent iteration:
        ujk += [1, -1, -1, -1, 1, -1]
        With probability p, flip the fourth '+1' (0-based indexing) to '-1'

    Parameters:
    n (int): Number of unit cells
    p (float): Probability to flip the fourth '+1'

    Returns:
    list: resulting ujk list
    """
    ujk = [1, 1, -1, -1, -1, -1]

    for _ in range(n - 1):
        next_segment = [1, 1, -1, -1, -1, -1]
        if np.random.rand() < p:
            # flip the fourth '+1', which is at index 4 (0-based)
            next_segment[4] = 1
        ujk += next_segment

    return np.array(ujk)




def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    

    # modified_lattice = diamond_chain_vertices(4)
    # bonds = diamond_chain_bonds(4)
    n = 100
    modified_lattice, coloring_solution = diamond_chain_lattice(n)
    print(modified_lattice.plaquettes)
    # target_flux = np.array([(-1) for p in modified_lattice.plaquettes], dtype=np.int8)
    
    ujk = construct_ujk(n, 0.5)[:-1] 
    check_fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)  #fluxes_from_bonds  fluxes_from_ujk




    fig, ax1 = plt.subplots(1, 1,  figsize=(10,8))  # 1 row 1 col
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    plot_edges(modified_lattice, ax= ax1, labels=coloring_solution, directions=None, lw=0.5, alpha=1)
    # plot_edges(modified_lattice, ax= ax1,labels=coloring_solution, directions=ujk)
    plot_plaquettes(modified_lattice, ax=ax1, labels = fluxes_to_labels(check_fluxes), color_scheme=np.array(['grey','w','wheat', 'thistle']))
    # plot_vertex_indices(modified_lattice, ax= ax1)

    plt.savefig("test_ham_figure.pdf", dpi=900,bbox_inches='tight')




    method = 'dense'
    data = diag_maj(modified_lattice, coloring_solution=None, ujk=ujk, method=method)
    # data = diag_maj_honeycomb(modified_lattice, coloring_solution, target_flux=target_flux, method=method, nnn=0.1)
    maj_energies = data['energies']
    maj_states = data['eigenvectors']


    # plot energy levels
    fig, ax = plt.subplots(1, 1,  figsize=(8,6))  # 1 row 1 col

    # DOS
    bandwidth = 0.02
    kde = gaussian_kde(maj_energies, bw_method=bandwidth)
    energy_min, energy_max = 0, maj_energies[-1]
    energy_range = np.arange(energy_min, energy_max, 0.0004)
    dos_values = kde(energy_range)
    dos_values[-1] = 0
    ax.plot(energy_range, dos_values / np.max(dos_values), lw=2, color='blue', label=r'{\rm DOS}(E)')
    # ax.set_xlim(0, 12)
    ax.set_ylim(ymin=0)
    ax.set_xlabel('E', fontsize=18)
    ax.set_ylabel(r'{\rm DOS}(E)', fontsize=18)
    ax.legend(loc='upper left', fontsize=18, frameon=False)
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)



    plt.savefig("diamond_dos.pdf", dpi=300,bbox_inches='tight')
    

    # ipr_values = data['ipr'][0, :]
    # assert(1 not in data['fluxes'])

    ground_state = np.abs(data['eigenvectors'][:, len(data['eigenvectors'])//2])**2
    site_index = get_closest_site_to_center(modified_lattice)
    # site_index = 0
    perturbed_state = perturb_ground_state(ground_state, site_index, perturbation_strength=100000000000) #0.3
    print(perturbed_state, max(perturbed_state))

    overlaps = data['eigenvectors'].conj().T @ perturbed_state
    overlaps_mag_sq = np.abs(overlaps)**2


    total_time = 10000000000000000 # 10000000000000000
    nframes = 20
    fps = nframes // 10 # total_time // nframes
    times = np.linspace(0, total_time, nframes)
    
    psi_t = time_evolution(maj_energies, maj_states, perturbed_state, times)
    # psi2 = np.abs(psi_t[:, 0])**2
    # plot_dist_smooth(modified_lattice, psi2, cmap="Purples", show_lattice=True, filename="time_evo.pdf")

    # loc_landscape = np.log(1/ np.abs(localization_landscape_dense(modified_lattice, coloring_solution, target_flux)))
    # plot_dist_smooth(modified_lattice,loc_landscape, vmin=-0.08, vmax=0.1, s=100, cmap='bwr', label="effective confinement potential", filename="time_evo.pdf")

    print(f"Wavefunctions shape: {psi_t.shape}")
    create_time_evolution_animation(modified_lattice, psi_t.T, total_time = total_time, cmap="Purples", target_flux=None, output_gif="time_evolution.gif", fps=fps)






if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
