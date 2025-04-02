import re
import math 
import sys
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
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import inv
from scipy.fft import fft, ifft

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


def Green(modified_lattice, target_flux, omega=0, broadening=0.01):
# constructing and solving majorana Hamiltonian for the gs sector
    print("Computing Green's function for E =", omega)
    ujk_init = np.full(modified_lattice.n_edges, -1)
    # J = np.array([1,1,1])
    ujk = ujk_from_fluxes(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    # fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)  #fluxes_from_bonds  fluxes_from_ujk

    n_vertices = modified_lattice.n_vertices
    # Initialize sparse Hamiltonian using LIL format for efficient incremental construction
    ham_sparse = lil_matrix((n_vertices, n_vertices), dtype=np.complex128)

    # Add NN terms
    for (i, j), u in zip(modified_lattice.edges.indices, ujk):
        ham_sparse[i, j] += u  # CW direction
        ham_sparse[j, i] -= u  # CCW direction

    # Assert particle-hole symmetry
    # ham_sparse_csr = ham_sparse.tocsr()
    # assert np.allclose(ham_sparse_csr.toarray(), -ham_sparse_csr.toarray().T, atol=1e-10), "Particle-hole symmetry is broken!"

    # Scale by the prefactor
    ham_sparse *= 0.5j
    ham_sparse.tocsr()


    I = sp.eye(n_vertices, dtype=complex, format="csc")
    factorized = spla.splu((omega + 1j * broadening) * I - ham_sparse)  # LU factorization
    return factorized.solve(I.toarray())  # Solve system instead of inversion



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


def compute_Gii(modified_lattice, target_flux, omega=0, broadening=0.01):
    """Compute the diagonal elements of the Green's function G_ii(E)"""
    z = omega + 1j * broadening
    G = Green(modified_lattice, target_flux, omega, broadening=0.01)  
    Gii = G.diagonal() 
    return Gii


def greens_function(energies, eigenvectors, psi_0, times):
    """
    Compute the Green's function G(t) = <psi0| exp(-iHt) | psi0>.

    Args:
        energies (np.ndarray): Eigenvalues of the Hamiltonian.
        eigenvectors (np.ndarray): Eigenvectors of the Hamiltonian.
        psi_0 (np.ndarray): Initial wavefunction.
        times (np.ndarray): Array of time points.

    Returns:
        np.ndarray: Green's function evaluated at each time t.
    """
    # Expand initial wavefunction in the eigenbasis:
    # coefficients: c_n = <eigenstate_n | psi_0>
    coefficients = np.dot(eigenvectors.T.conj(), psi_0)

    # Precompute exponential factors for each eigenstate and each time:
    # evolution_factors has shape (n_eigenstates, n_times)
    evolution_factors = np.exp(-1j * np.outer(energies, times))

    # Calculate the Green's function:
    # G(t) = sum_n |c_n|^2 e^{-iE_n t}
    greens = np.dot(np.abs(coefficients)**2, evolution_factors)
    return greens



def greens_function_local_diagonals(energies, eigenvectors, t):
    """
    Compute the diagonal elements of the Green's function matrix in the real-space basis
    for a fixed time t.
    
    The Green's function is defined as:
        G_{ij}(t) = <i| exp(-i H t) |j>
    
    and its diagonal elements are:
        G_{ii}(t) = sum_n |<i|n>|^2 * exp(-i E_n t)
    
    Args:
        energies (np.ndarray): 1D array of eigenenergies (E_n), shape (n_eigenstates,).
        eigenvectors (np.ndarray): 2D array of eigenvectors with shape 
                                   (n_local_states, n_eigenstates). 
                                   Each column corresponds to an eigenstate.
        t (float): The time at which to evaluate the Green's function.
    
    Returns:
        np.ndarray: 1D array of diagonal elements G_{ii}(t) for each local state i.
    """
    # Compute the time evolution factor for each eigenstate at time t:
    exp_factors = np.exp(-1j * energies * t)  # shape: (n_eigenstates,)
    
    # Compute the probability weights |<i|n>|^2 for each local state i and eigenstate n:
    prob_amplitudes = np.abs(eigenvectors)**2  # shape: (n_local_states, n_eigenstates)
    
    # Multiply each eigenstate contribution by its time evolution factor and sum over eigenstates:
    # This gives the diagonal element for each local state i.
    G_diag = np.sum(prob_amplitudes * exp_factors, axis=1)
    return G_diag


def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    
    # modified_lattice, coloring_solution = honeycomb_lattice(40, return_coloring=True)
    L = 43
    modified_lattice = n_ladder(L)    # print(modified_lattice.vertices)


    total_plaquettes = len(modified_lattice.plaquettes)

    flux_filling = 0.5
    
    num_seed = 100
    Tmax = 100
    dt = 1

    Rs_coherent = np.zeros((num_seed, Tmax // dt), dtype=np.float64)
    Rs_ensemble = np.zeros((num_seed, Tmax // dt), dtype=np.float64)

    for seed in range(num_seed):
        print("Seed = ", seed)
        target_flux = flux_sampler(modified_lattice, int(total_plaquettes * flux_filling), seed = seed)  # 4434
        print("Total plaquettes = ", total_plaquettes)
        print("Total sites = ", modified_lattice.n_vertices)
        print("Flux filling = ", flux_filling)
        # print(target_flux)



        method = 'dense'
        data = diag_maj(modified_lattice, coloring_solution=None, target_flux=target_flux, method=method)
        maj_energies = data['energies']
        maj_eigenvectors = data['eigenvectors']
        # assert(1 not in data['fluxes'])
        print(data['fluxes'])
        print(complex_fluxes_to_labels(data['fluxes']))


        R_coherent = np.zeros((Tmax//dt, 2), dtype=np.float64)
        R_ensemble = np.zeros((Tmax//dt, 2), dtype=np.float64)

        for i, times in enumerate(np.arange(0, Tmax, dt)):
            G_diag_t = greens_function_local_diagonals(maj_energies, maj_eigenvectors, times)
            Rt_squared = np.sum(np.abs(G_diag_t) ** 2) 
            Rt = np.sum(G_diag_t) 

            R_coherent[i, 0] = times
            R_coherent[i, 1] = Rt
            R_ensemble[i, 0] = times
            R_ensemble[i, 1] = Rt_squared

        R_coherent[:, 1] = R_coherent[:,1] / modified_lattice.n_vertices
        R_ensemble[:, 1] = R_ensemble[:,1] / modified_lattice.n_vertices

        Rs_coherent[seed, :] = R_coherent[:, 1]
        Rs_ensemble[seed, :] = R_ensemble[:, 1]


    R_ensemble = np.mean(Rs_ensemble, axis=0)
    R_coherent = np.abs(np.mean(Rs_coherent, axis=0))** 2
    # print(R_coherent)
    
    R_main_coherent = np.sum(R_coherent[10::]) / len(R_coherent[10::])
    R_main_ensemble = np.sum(R_ensemble[10::]) / len(R_ensemble[10::])


    print(f"R_main_coherent = {R_main_coherent}")
    print(f"R_main_ensemble = {R_main_ensemble}")


    



    # Plot return probability in log-log scale
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(0, Tmax, dt), R_ensemble, '-', color='red', linewidth=2)
    plt.plot(np.arange(0, Tmax, dt), R_coherent, '-', color='blue', linewidth=0.8)
    plt.hlines(y=R_main_ensemble, xmin=0, xmax=Tmax, color='red', linestyle='--', linewidth=0.5)    
    plt.hlines(y=R_main_coherent, xmin=0, xmax=Tmax, color='blue', linestyle='--', linewidth=0.5)

    plt.xlabel("tJ")
    plt.ylabel("$R(t)$")
    plt.xscale('log')
    # plt.yscale('log')
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.ylim(ymin=min(R_coherent)*0.5) 
    plt.ylim(ymin=0,ymax=0.2)
    plt.savefig("P_RETURN_TIME.pdf", dpi=300, bbox_inches='tight')


    
        


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

