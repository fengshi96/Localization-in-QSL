import re
import math 
import sys, os
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from numpy.random import default_rng
import scipy
import primme
import random
from collections import Counter
import numpy.typing as npt
from koala.pointsets import uniform
from koala.voronization import generate_lattice
from koala.example_graphs import higher_coordination_number_example
from koala.plotting import plot_edges, plot_vertex_indices, plot_lattice, plot_plaquettes
from koala.graph_utils import vertices_to_polygon, make_dual
from koala.graph_color import color_lattice, edge_color
from koala.flux_finder import fluxes_from_bonds, fluxes_from_ujk, fluxes_to_labels, ujk_from_fluxes, n_to_ujk_flipped, find_flux_sector
from koala.example_graphs import make_amorphous, ground_state_ansatz, single_plaquette, honeycomb_lattice, n_ladder
import koala.hamiltonian as ham
from koala.lattice import Lattice, cut_boundaries
from koala import chern_number as cn
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial import Voronoi, voronoi_plot_2d

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def single_vertex():
    vertices = np.array([
        [0, 0], 
        [1, 0], 
        [0.5, np.sqrt(3)/2], 
        [0.5, np.sqrt(3)/4],
    ]) * 1 + np.array([0.0, 0.05])

    edge_indices = np.array([
        [4, 1], 
        [4, 2], 
        [4, 3]
    ]) - 1

    edge_crossing = np.zeros_like(edge_indices)
    lattice = Lattice(vertices, edge_indices, edge_crossing)
    return lattice



def diag_maj(modified_lattice, coloring_solution, target_flux, method='dense', k=1, max_ipr_level=5):
# constructing and solving majorana Hamiltonian for the gs sector
    ujk_init = np.full(modified_lattice.n_edges, -1)
    J = np.array([1,1,1])
    # ujk = find_flux_sector(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    # fluxes = fluxes_from_bonds(modified_lattice, ujk, real=False)  #fluxes_from_bonds  fluxes_from_ujk
    ujk = ujk_from_fluxes(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)  #fluxes_from_bonds  fluxes_from_ujk

    if coloring_solution is not None:
        maj_ham = ham.majorana_hamiltonian(modified_lattice, coloring_solution, ujk)
    else:
        maj_ham = ham.majorana_hamiltonian(modified_lattice, None, ujk)
    
    if method == 'dense':
        maj_energies, eigenvectors = scipy.linalg.eigh(maj_ham)
    else:
        smaj_ham = sparse.csr_matrix(maj_ham)
        # maj_energies, eigenvectors = scipy.sparse.linalg.eigs(smaj_ham, k=1, which='SM')
        maj_energies, eigenvectors = primme.eigsh(smaj_ham, k=k, tol=1e-10, which=0)

    gap = min(np.abs(maj_energies))
    ipr_values = np.array([(np.sum(np.abs(eigenvectors)**(2*q), axis=0)) for q in np.arange(2, max_ipr_level+1, 1)])
    # print("shape of IPR matrix", ipr_values.shape)
    print('Gap =', gap)

    epsilon = 1e-8
    num_zero_energy_levels = np.sum(np.abs(maj_energies) < epsilon)
    print(f"Zero-energy levels: {num_zero_energy_levels}")

    data = {}
    data['gap'] = gap
    data['ipr'] = ipr_values
    data['ujk'] = ujk
    data['fluxes'] = fluxes   # for checking if solved ujk produces the desired flux pattern
    data['energies'] = maj_energies
    data['eigenvectors'] = eigenvectors


    return data



def diag_maj_honeycomb(modified_lattice, target_flux, nnn=0.0, max_ipr_level = 5, method='dense', k=1, which='SA'):
# constructing and solving majorana Hamiltonian for the gs sector
    ujk_init = np.full(modified_lattice.n_edges, -1)
    J = np.array([1,1,1])
    ujk = ujk_from_fluxes(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)  #fluxes_from_bonds  fluxes_from_ujk

    
    if method == 'dense':
        maj_ham = majorana_hamiltonian_with_nnn(modified_lattice, ujk, nnn_strength=nnn)
        maj_energies, eigenvectors = scipy.linalg.eigh(maj_ham)
    else:
        # smaj_ham = csr_matrix(maj_ham)
        smaj_ham = sparse_majorana_hamiltonian_with_nnn(modified_lattice, ujk, nnn_strength=nnn)
        # maj_energies, eigenvectors = scipy.sparse.linalg.eigs(smaj_ham, k=1, which='SM')
        maj_energies, eigenvectors = primme.eigsh(smaj_ham, k=k, tol=1e-12, which=which)

    gap = min(np.abs(maj_energies))
    ipr_values = np.array([(np.sum(np.abs(eigenvectors)**(2*q), axis=0)) for q in np.arange(2, max_ipr_level+1, 1)])
    # print("shape of IPR matrix", ipr_values.shape)
    print('Gap =', gap)

    epsilon = 1e-8
    num_zero_energy_levels = np.sum(np.abs(maj_energies) < epsilon)
    print(f"Zero-energy levels: {num_zero_energy_levels}")

    data = {}
    data['gap'] = gap
    data['ipr'] = ipr_values
    data['ujk'] = ujk
    data['fluxes'] = fluxes   # for checking if solved ujk produces the desired flux pattern
    data['energies'] = maj_energies
    data['eigenvectors'] = eigenvectors


    return data


def majorana_hamiltonian_with_nnn(
    lattice: Lattice,
    ujk: npt.NDArray,
    J: npt.NDArray[np.floating] = np.array([1.0, 1.0, 1.0]),
    nnn_strength: float = 0.1,
) -> npt.NDArray[np.complexfloating]:
    """
    Construct the Majorana Hamiltonian for the Kitaev model on a honeycomb lattice,
    including next-nearest-neighbor (NNN) couplings.

    Args:
        lattice (Lattice): Lattice object representing the honeycomb lattice.
        ujk (npt.NDArray): Link variables for nearest-neighbor interactions (+1 or -1).
        J (npt.NDArray[np.floating]): Coupling constants for nearest neighbors.
        nnn_strength (float): Strength of the NNN coupling perturbation.

    Returns:
        npt.NDArray[np.complexfloating]: Quadratic Majorana Hamiltonian matrix.
    """
    # Initialize Hamiltonian
    ham = np.zeros((lattice.n_vertices, lattice.n_vertices), dtype=np.complex128)
    # Add NN terms
    ham[lattice.edges.indices[:, 1], lattice.edges.indices[:, 0]] = ujk
    ham[lattice.edges.indices[:, 0], lattice.edges.indices[:, 1]] = -1 * ujk

    # Next-nearest-neighbor (NNN) couplings
    for plaquette in lattice.plaquettes:
        # Use the CCW ordered vertices directly
        vertices = plaquette.vertices
        # print(vertices)
        # Define the NNN pairs based on CCW ordering
        nnn_pairs = [
            (vertices[0], vertices[4]),
            (vertices[4], vertices[2]),
            (vertices[2], vertices[0]),
            (vertices[1], vertices[5]),
            (vertices[5], vertices[3]),
            (vertices[3], vertices[1]),
        ]

        # print(nnn_pairs)

        for v1, v2 in nnn_pairs:
            ham[v1, v2] += nnn_strength  # CW direction: +1
            ham[v2, v1] -= nnn_strength  # CCW direction: -1

    assert np.allclose(ham, -ham.T, atol=1e-10), "Particle-hole symmetry is broken!"
    ham = ham * 2.0j 

    return ham

def sparse_majorana_hamiltonian_with_nnn(
    lattice: Lattice,
    ujk: npt.NDArray,
    nnn_strength: float = 0.1,
) -> csr_matrix:
    """
    Construct the Majorana Hamiltonian for the Kitaev model on a honeycomb lattice,
    including next-nearest-neighbor (NNN) couplings, as a sparse matrix.

    Args:
        lattice (Lattice): Lattice object representing the honeycomb lattice.
        ujk (npt.NDArray): Link variables for nearest-neighbor interactions (+1 or -1).
        nnn_strength (float): Strength of the NNN coupling perturbation.

    Returns:
        csr_matrix: Quadratic Majorana Hamiltonian matrix in sparse format.
    """
    n_vertices = lattice.n_vertices
    # Initialize sparse Hamiltonian using LIL format for efficient incremental construction
    ham_sparse = lil_matrix((n_vertices, n_vertices), dtype=np.complex128)

    # Add NN terms
    for (i, j), u in zip(lattice.edges.indices, ujk):
        ham_sparse[i, j] += u  # CW direction
        ham_sparse[j, i] -= u  # CCW direction

    # Add NNN terms
    for plaquette in lattice.plaquettes:
        vertices = plaquette.vertices
        nnn_pairs = [
            (vertices[0], vertices[4]),
            (vertices[4], vertices[2]),
            (vertices[2], vertices[0]),
            (vertices[1], vertices[5]),
            (vertices[5], vertices[3]),
            (vertices[3], vertices[1]),
        ]

        for v1, v2 in nnn_pairs:
            ham_sparse[v1, v2] += nnn_strength  # CW direction
            ham_sparse[v2, v1] -= nnn_strength  # CCW direction

    # Assert particle-hole symmetry
    ham_sparse_csr = ham_sparse.tocsr()
    assert np.allclose(ham_sparse_csr.toarray(), -ham_sparse_csr.toarray().T, atol=1e-10), "Particle-hole symmetry is broken!"

    # Scale by the prefactor
    ham_sparse *= 2.0j

    # Convert to CSR format for efficient matrix operations
    return ham_sparse.tocsr()


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

def complex_fluxes_to_labels(fluxes: np.ndarray) -> np.ndarray:
    """
    Auxilliary function to plot complex fluxes
    Remaps fluxes from the set {1,-1, +i, -i} to labels in the form {0,1,2,3} for plotting.
    Args:
        fluxes (np.ndarray): Fluxes in the format +1 or -1 or +i or -i
    Returns:
        np.ndarray: labels in [0(+1),1(-1),2(+i),3(-i)], to which I later assign the color_scheme=np.array(['w','lightgrey','wheat', 'thistle'])
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

def plot_wave_function_smooth(lattice, wave_function_distribution, resolution=300):
    """
    Plot the wave function |ψ_i|^2 distribution on the lattice with a smooth and transparent scatter style.

    Args:
        lattice (Lattice): The lattice object.
        wave_function_distribution (np.ndarray): Probability distribution |ψ_i|^2.
    """
    # Extract vertex positions
    positions = lattice.vertices.positions
    x, y = positions[:, 0], positions[:, 1]

    # Normalize the wave function distribution
    normalized_distribution = wave_function_distribution / np.max(wave_function_distribution)

    # Create the plot
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        x, y,
        c=normalized_distribution,  # Color by normalized |ψ_i|^2
        cmap='plasma',              # Smooth colormap
        s=20,                      # Size of the scatter points
        alpha=0.4,                  # Transparency
        edgecolors='none'           # Remove edge colors for smoothness
    )
    plt.colorbar(scatter, label=r"$|\psi_i|^2$")
    plt.title("Wave Function Distribution on Lattice (Smooth Scatter)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    
    # modified_lattice, coloring_solution = honeycomb_lattice(40, return_coloring=True)
    L = 3000
    modified_lattice = n_ladder(L)    # print(modified_lattice.vertices)
    # print(modified_lattice.edges)

    # target_flux = np.array(
    #     [ground_state_ansatz(p.n_sides) for p in modified_lattice.plaquettes],
    #     dtype=np.int8)
    
    # target_flux = np.array(
    #     [(-1) for p in modified_lattice.plaquettes],
    #     dtype=np.int8)

    total_plaquettes = len(modified_lattice.plaquettes)

    Dos_E0 = []
    Dos_all = []
    Ipr = []
    Eng = []

    flux_filling = 0.5

    for i in range(1, 5):
        target_flux = flux_sampler(modified_lattice, int(total_plaquettes * flux_filling), seed = i)  # 4434
        print("Total plaquettes = ", total_plaquettes)
        print("Total sites = ", modified_lattice.n_vertices)
        print("Flux filling = ", flux_filling)
        # print(target_flux)


        method = 'dense'
        data = diag_maj(modified_lattice, coloring_solution=None, target_flux=target_flux, method=method)
        # data = diag_maj_honeycomb(modified_lattice, target_flux, nnn=0.2, method='dense')
        maj_energies = data['energies']
        ipr_values = data['ipr'][0, :]
        # assert(1 not in data['fluxes'])
        print(data['fluxes'])
        print(complex_fluxes_to_labels(data['fluxes']))

        # print(ipr_values[int(len(ipr_values)//2)])
        # print(int(len(ipr_values)//2))
        # print(maj_energies[int(len(ipr_values)//2)])

        bandwidth = 0.02
        kde = gaussian_kde(maj_energies, bw_method=bandwidth)
        energy_min, energy_max = 0, maj_energies[-1]
        energy_range = np.linspace(energy_min, energy_max, 1000)
        dos_values = kde(energy_range)
        print("DOS at zero energy:", dos_values[0])

        Dos_E0.append(dos_values[0])
        Dos_all.append(dos_values)
        Ipr.append(ipr_values)
        Eng.append(maj_energies)
    
        if method != 'sparse':
            if i % 10 == 0:
                # plot energy levels
                fig, ax = plt.subplots(1, 3,  figsize=(30,10))  # 1 row 1 col
                ax[0].set_title('Energy Levels')
                ax[0].scatter(range(len(maj_energies)), maj_energies)
                ax[0].hlines(y=0, xmin=0, xmax=len(maj_energies), linestyles='dashed')
                ax[0].set_xlabel('Energy Level Index', fontsize=18)
                ax[0].set_ylabel('Energy', fontsize=18)
                ax[0].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)

                # DOS
                bandwidth = 0.02
                kde = gaussian_kde(maj_energies, bw_method=bandwidth)
                energy_min, energy_max = 0, maj_energies[-1]
                energy_range = np.linspace(energy_min, energy_max, 1000)
                dos_values = kde(energy_range)
                print(energy_range[:10], dos_values[:10])
                ax[1].plot(energy_range, dos_values, lw=2)
                ax[1].set_xlabel('Energy', fontsize=18)
                ax[1].set_ylabel('DOS', fontsize=18)
                ax[1].set_title('Density of States (DOS)')
                ax[1].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)

                # Plot the IPR as a function of energy
                energy_range_min = 0.0
                energy_range_max = 1.5
                filtered_indices = np.where((maj_energies >= energy_range_min) & (maj_energies <= energy_range_max))[0]
                filtered_energies = maj_energies[filtered_indices]
                filtered_ipr = ipr_values[filtered_indices]
                ax[2].scatter(filtered_energies, filtered_ipr)
                ax[2].set_xlabel('Energy', fontsize=18)
                ax[2].set_ylabel('IPR', fontsize=18)
                ax[2].set_title('IPR vs Energy')
                ax[2].set_yscale('log')
                ax[2].set_ylim(ymax=1e-0, ymin=1e-4)
                ax[2].set_xlim(energy_range_min, energy_range_max)
                ax[2].grid(False)
                ax[2].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)

    if not os.path.exists("./data/"):
        os.makedirs("./data/")
    plt.savefig("./data/" + "dos_F"+str(flux_filling) + "_S" + str(i) + ".pdf", dpi=300,bbox_inches='tight')

    Ipr = np.array(Ipr)
    Dos_E0 = np.array(Dos_E0)
    Dos_all = np.array(Dos_all)
    Eng = np.array(Eng)

    Dos_E0_mean = np.mean(Dos_E0)
    Dos_all_mean = np.mean(Dos_all, axis=0)
    Ipr_mean = np.mean(Ipr, axis=0)
    Eng_mean = np.mean(Eng, axis=0)

    print("mean DOS at zero energy:", Dos_E0_mean)
    print("mean DOS at all energies:", Dos_all_mean)
    print("mean IPR at all energies:", Ipr_mean)
    print("mean Energies:", Eng_mean)

    printfArray(Dos_E0_mean, "DOS0_mean_"+str(L)+".dat")
    printfArray(Ipr_mean, "IPR_mean_"+str(L)+".dat")
    printfArray(Eng_mean, "ENG_mean_"+str(L)+".dat", transpose=True)
    printfArray(Dos_all_mean, "DOS_all_mean_"+str(L)+".dat", transpose=True)

    # data for each random seed
    printfArray(Dos_E0, "DOS0_"+str(L)+".dat")
    printfArray(Ipr, "IPR_"+str(L)+".dat")
    printfArray(Eng, "ENG_"+str(L)+".dat", transpose=True)
    printfArray(Dos_all, "DOS_all_"+str(L)+".dat", transpose=True)

    # fig, ax = plt.subplots(1, 1,  figsize=(8,6))
    # ax.plot(np.arange(0, 0.6, 0.01), Dos)
    # plt.savefig("DOS_L"+str(L)+"_S"+str(i)+".pdf", dpi=300,bbox_inches='tight')


def printfArray(A, filename, transpose = False):
    file = open(filename, "w")
    try:
        col = A.shape[1]
    except IndexError:
        A = A.reshape(-1, 1) 
    
    row = A.shape[0]
    col = A.shape[1]

    if transpose == False:
        for i in range(row):
            for j in range(col - 1):
                file.write(str(A[i, j]) + " ")
            file.write(str(A[i, col - 1]))  # to avoid whitespace at the end of line
            file.write("\n")
    elif transpose == True:
        for i in range(col):
            for j in range(row - 1):
                file.write(str(A[j, i]) + " ")
            file.write(str(A[row - 1, i]))
            file.write("\n")
    else:
        raise ValueError("3rd input must be Bool")
    file.close()



if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)

