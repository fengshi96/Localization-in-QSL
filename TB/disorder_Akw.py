import sys
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
mpl.rc('text', usetex=True)


def h_clean(k, lam=0.0):
    Jx = Jy = Jz = 1.0
    M = np.zeros((4,4), dtype=np.complex128)
    # z rungs
    M[0,1] =  1j*Jz;  M[1,0] = -1j*Jz
    M[2,3] =  1j*Jz;  M[3,2] = -1j*Jz
    # intra-cell x,y
    M[0,2] =  1j*Jx;  M[2,0] = -1j*Jx
    M[3,1] =  1j*Jy;  M[1,3] = -1j*Jy
    # inter-cell x,y with e^{±ik}
    ek   = np.exp(1j*k)
    emk  = np.exp(-1j*k)
    M[3,1] +=  1j*Jy*ek;   M[1,3] += -1j*Jy*emk
    M[0,2] +=  1j*Jx*emk;  M[2,0] += -1j*Jx*ek

    # ——— diagonal NNN “loops” ———
    s = np.sin(k)
    M[0,0] = +2 * lam * s
    M[1,1] = -2 * lam * s
    M[2,2] = -2 * lam * s
    M[3,3] = +2 * lam * s

    return M * 0.5



def gidx(cell, site):
    """flatten (cell, site) → matrix index"""
    return 4 * cell + site


# def majorana_disordered(L, p_flip=0.5, seed=None,
#                         Jx=1.0, Jy=1.0, Jz=1.0):
#     """
#     Dense 4L×4L quadratic Majorana Hamiltonian.
#     Each z-rung is multiplied by −1 with probability p_flip.
#     """
#     if seed is not None:
#         np.random.seed(seed)

#     H = np.zeros((4 * L, 4 * L), dtype=np.complex128)

#     for r in range(L):
#         rp = (r + 1) % L    # cell to the right

#         # ---- z rungs: 0↔1  and 2↔3 ----
#         sgn01 = -1 if np.random.rand() < p_flip else 1
#         sgn23 = -1 if np.random.rand() < p_flip else 1


#         H[gidx(r, 0), gidx(r, 1)] =  1j * sgn01 * Jz
#         H[gidx(r, 1), gidx(r, 0)] = -1j * sgn01 * Jz

#         H[gidx(r, 2), gidx(r, 3)] =  1j * sgn23 * Jz
#         H[gidx(r, 3), gidx(r, 2)] = -1j * sgn23 * Jz

#         # ---- intra-cell x/y ----
#         H[gidx(r, 0), gidx(r, 2)] =  1j * Jx 
#         H[gidx(r, 2), gidx(r, 0)] = -1j * Jx 

#         H[gidx(r, 3), gidx(r, 1)] =  1j * Jy 
#         H[gidx(r, 1), gidx(r, 3)] = -1j * Jy 

#         # ---- inter-cell x/y (PBC) ----
#         H[gidx(r, 3), gidx(rp, 1)] =  1j * Jy 
#         H[gidx(rp, 1), gidx(r, 3)] = -1j * Jy 

#         H[gidx(rp, 0), gidx(r, 2)] =  1j * Jx
#         H[gidx(r, 2), gidx(rp, 0)] = -1j * Jx

#     return H * 0.5


def majorana_disordered(L, p_flip=0.5, seed=None,
                        Jx=1.0, Jy=1.0, Jz=1.0, lam=0.0):
    """
    4L×4L Majorana Hamiltonian with
      - random z‐rung flips (prob p_flip),
      - NN couplings Jx,Jy,Jz,
      - NNN loops of strength lam (no u’s).
    """
    if seed is not None:
        np.random.seed(seed)

    N = 4*L
    H = np.zeros((N, N), dtype=np.complex128)

    # 1) build NN part 
    for r in range(L):
        rp = (r + 1) % L

        # z‐rungs
        s01 = -1 if np.random.rand() < p_flip else +1
        i, j = gidx(r,0), gidx(r,1)
        H[i,j] =  1j*s01*Jz;  H[j,i] = -1j*s01*Jz
        s23 = -1 if np.random.rand() < p_flip else +1
        i, j = gidx(r,2), gidx(r,3)
        H[i,j] =  1j*s23*Jz;  H[j,i] = -1j*s23*Jz

        # intra‐cell x/y
        i, j = gidx(r,0), gidx(r,2)
        H[i,j] =  1j*Jx;     H[j,i] = -1j*Jx
        i, j = gidx(r,3), gidx(r,1)
        H[i,j] =  1j*Jy;     H[j,i] = -1j*Jy

        # inter‐cell x/y
        i, j = gidx(r,3),  gidx(rp,1)
        H[i,j] =  1j*Jy;     H[j,i] = -1j*Jy
        i, j = gidx(rp,0), gidx(r,2)
        H[i,j] =  1j*Jx;     H[j,i] = -1j*Jx


    return H * 0.5




def sample_flux_sites(L, filling, seed=None):
    rng = np.random.default_rng(seed)
    corners = np.arange(0, 4*L, 2)            # 0,2,4,…,4L−2
    M       = int(np.rint(filling * corners.size))
    if M % 2 == 1:
        M += 1 if M < corners.size else -1    # make M even
    M = max(0, min(M, corners.size))
    return np.sort(rng.choice(corners, size=M, replace=False))



def majorana_disordered_B(L, filling=0.5, seed=None,
                          Jx=1.0, Jy=1.0, Jz=1.0):
    """
    4L×4L Majorana H with exactly `filling` fraction of plaquette-fluxes
    flipped by toggling all z-rungs in the cell-range between each chosen
    pair of corners.
    """
    # 1) sample even corner sites and pair them
    flux_corners = sample_flux_sites(L, filling, seed)
    print(flux_corners)

    N = 4*L
    H = np.zeros((N, N), dtype=np.complex128)
    def gidx(r,s): return 4*r + s

    # 2) build the ground-state sector
    for r in range(L):
        rp = (r+1) % L
        # vertical z-rungs
        H[gidx(r,0),gidx(r,1)] =  1j*Jz
        H[gidx(r,1),gidx(r,0)] = -1j*Jz
        H[gidx(r,2),gidx(r,3)] =  1j*Jz
        H[gidx(r,3),gidx(r,2)] = -1j*Jz
        # x/y bonds
        H[gidx(r,0),gidx(r,2)] =  1j*Jx; H[gidx(r,2),gidx(r,0)] = -1j*Jx
        H[gidx(r,3),gidx(r,1)] =  1j*Jy; H[gidx(r,1),gidx(r,3)] = -1j*Jy
        H[gidx(r,3),gidx(rp,1)] =  1j*Jy; H[gidx(rp,1),gidx(r,3)] = -1j*Jy
        H[gidx(rp,0),gidx(r,2)] =  1j*Jx; H[gidx(r,2),gidx(rp,0)] = -1j*Jx

    # 3) for each sampled pair (a,b), flip every cell in [a//4 .. b//4]
    for i in range(0, len(flux_corners), 2):
        a, b = flux_corners[i], flux_corners[i+1]
        cell_start = a // 4
        cell_end   = b // 4
        print(f"a = {a}, b = {b}, cell = {cell_start}..{cell_end}")
        for cell in range(cell_start, cell_end+1):
            # flip both z-rungs in this cell
            for (s1,s2) in [(0,1),(2,3)]:
                i1, i2 = gidx(cell,s1), gidx(cell,s2)
                # print(f"i1 = {i1}, i2 = {i2}")
                H[i1, i2] *= -1
                H[i2, i1] *= -1

    # 4) include the 1/2 Majorana prefactor
    return 0.5*H


def spectral_function_B(L, filling=0, seeds=1,
                      eta=0.01, w_max=1.5, dw=0.005, Jx=1.0, Jy=1.0, Jz=1.0, lam=0.0):
    U, k_vals = bloch_basis_matrix(L)
    nk  = len(k_vals)
    w_vals = np.arange(-w_max, w_max+dw/2, dw)
    nw  = len(w_vals)
    A   = np.zeros((nk, nw))

    lor = lambda x: eta/np.pi/(x**2 + eta**2)

    for sd in seeds:
        print(f"seed = {sd}, filling = {filling}")
        H  = majorana_disordered_B(L, filling, sd)
        C, E = coefficients(H, U)
        # reshape C to (nk,4,n)   → sum over α in axis=1
        C2 = np.abs(C).reshape(nk,4,-1)**2  # square modulus
        Wk_n = C2.sum(axis=1)               # shape (nk , n_states)

        for n, En in enumerate(E):
            A += Wk_n[:,n,None] * lor(w_vals-En)

    return A/len(seeds), k_vals, w_vals



def plot_bands(L, k_vals):
    bands  = np.zeros((4, len(k_vals)), dtype=np.float64)

    for j, k in enumerate(k_vals):
        bands[:, j] = np.sort(la.eigvalsh(h_clean(k)).real)

    plt.figure(figsize=(5,3))
    for a in range(4):
        plt.plot(k_vals, bands[a], lw=1.5)
    plt.axhline(0, color='k', lw=.5)
    plt.xlim(-np.pi, np.pi)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\varepsilon_\alpha(k)$')
    plt.title(f'4-site Kitaev ladder, L={L}')
    plt.tight_layout()
    plt.show()



def compute_coefficients(Hdis, U, n_states=None):
    """
    Return the coefficient matrix  C = U† V.

    Parameters
    ----------
    Hdis : (4L, 4L) complex ndarray
        Dense real-space Majorana Hamiltonian for *one*
        disorder realisation.
    U    : (4L, 4L) complex ndarray
        Clean Bloch basis with columns |α,k⟩ arranged as
        col = 4*k_index + α  (α = 0..3, k_index = 0..L-1).
    n_states : int or None
        If given, keep only the first `n_states` eigen-vectors
        of Hdis (useful when you need, e.g., only the negative-
        energy sector).  Default = 4L (all states).

    Returns
    -------
    C : complex ndarray, shape (4L, n_states)
        Matrix of coefficients  C[4*k+α , n] = c^{n}_{α,k}.
    E : 1-D real ndarray, length n_states
        Corresponding eigen-energies E_n (sorted).
    """
    # --- diagonalise Hdis ---
    Evals, V = la.eigh(Hdis)          # V columns = φ_n

    # optionally restrict to the first n_states eigen-pairs
    if n_states is not None:
        Evals = Evals[:n_states]
        V     = V[:, :n_states]

    # --- projection C = U† V ---
    C = U.conj().T @ V
    return C, Evals


def bloch_basis_matrix(L, lam=0.0):
    k_vals = 2*np.pi*np.arange(-L/2, L/2)/L
    U = np.zeros((4*L, 4*L), dtype=np.complex128)
    for k_idx,k in enumerate(k_vals):
        u = la.eigh(h_clean(k, lam))[1].T             # shape (4,4)
        for alpha in range(4):
            col = 4*k_idx + alpha
            for r in range(L):
                phase = np.exp(1j*k*r)/np.sqrt(L)
                U[4*r:4*r+4, col] = u[alpha]*phase
    return U, k_vals


def coefficients(H, U):
    E, V = la.eigh(H)
    C = U.conj().T @ V
    return C, E


def spectral_function(L, p_flip, seeds,
                      eta=0.01, w_max=1.5, dw=0.005, Jx=1.0, Jy=1.0, Jz=1.0, lam=0.0):
    U, k_vals = bloch_basis_matrix(L, lam)
    nk  = len(k_vals)
    w_vals = np.arange(-w_max, w_max+dw/2, dw)
    nw  = len(w_vals)
    A   = np.zeros((nk, nw))

    lor = lambda x: eta/np.pi/(x**2 + eta**2)

    for sd in seeds:
        print(f"seed = {sd}, p_flip = {p_flip}")
        H  = majorana_disordered(L, p_flip, sd, Jx, Jy, Jz, lam)
        C, E = coefficients(H, U)
        # reshape C to (nk,4,n)   → sum over α in axis=1
        C2 = np.abs(C).reshape(nk,4,-1)**2  # square modulus
        Wk_n = C2.sum(axis=1)               # shape (nk , n_states)

        for n, En in enumerate(E):
            A += Wk_n[:,n,None] * lor(w_vals-En)

    return A/len(seeds), k_vals, w_vals



def main(total, cmdargs):

    L      = 20          # unit cells
    p_flip = 0.0  # float(cmdargs[1])      # probability to flip a z rung

    w_max  = 0.2         # |ω| range shown
    w_full = 1.0         # |ω| range for the spectral function

    seeds  = range(50)      # 50 disorder realisations

    Aavg, k_vals, w_vals = spectral_function(L, p_flip, seeds, Jx=1.0, Jy=1.0, Jz=1.0, lam=0.8)
    Aavg_log = np.log10(Aavg)



    K,W = np.meshgrid(k_vals, w_vals, indexing='ij')
    fig, ax = plt.subplots(1, 2,  figsize=(12,6))

    levels = np.linspace(-0.5, 0.8, 20)
    ax[0].contourf(K, W, Aavg_log, levels=levels, cmap='jet', extend="both")
    ax[0].contourf(K, W, Aavg_log, levels=levels, cmap='jet', extend="both")

    cf = ax[1].contourf(K, W, Aavg_log, levels=levels, cmap='jet', extend="both")
    cf = ax[1].contourf(K, W, Aavg_log, levels=levels, cmap='jet', extend="both")



    # cbar = ax.colorbar()
    cax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cf,
                        ax=ax.tolist(),    # or ax=axes, either works
                        orientation='vertical',
                        pad=0.02,            # space between plot and colorbar
                        fraction=0.05,  # width of colorbar
                        cax=cax)       

    cbar.set_label(r"$\log_{10}(A(k,\omega))$", rotation=270, labelpad=15)


    ax[0].set_xlabel('$k$', fontsize=20)
    ax[1].set_xlabel('$k$', fontsize=20)
    ax[0].set_ylabel(r'$\omega$', fontsize=20)
    ax[1].set_ylabel(r'$\omega$', fontsize=20)
    ax[0].set_ylim(0, w_full)
    ax[1].set_ylim(0, w_max)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[1].tick_params(axis='x', labelsize=20)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[1].tick_params(axis='y', labelsize=20)
    ax[1].set_yticks([0.0, 0.1, 0.2])

    ax[0].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi] - (k_vals[1] - k_vals[0]) / 2)
    ax[0].set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax[1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi] - (k_vals[1] - k_vals[0]) / 2)
    ax[1].set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    # plt.title(f'Disorder average  (L={L}, p={p_flip}, {len(seeds)} samples)')

    fig.suptitle(
        rf'\textbf{{Disorder average}} '
        rf'$(L={L},\;p={p_flip:.3f},\;{len(seeds)}\ \mathrm{{samples}})$',
        fontsize=20,
        y=1.02,
        ha='center'
    )
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./Akw_avg_{p_flip:.3f}.pdf", dpi=200,bbox_inches='tight')




    # dk   = k_vals[1] - k_vals[0]          # spacing in k grid
    # Sw  = Aavg.sum(axis=0) * dk          # shape (nw,)

    # fig2, ax2 = plt.subplots(figsize=(8,6))
    # ax2.plot(w_vals, Sw, lw=1.8)
    # ax2.set_xlabel(r'$\omega$')
    # ax2.set_ylabel(r'$S(\omega)$')
    # ax2.set_title('k–integrated spectral function')
    # ax2.set_xlim(0, w_max)
    # plt.tight_layout()
    # plt.savefig(f"Aw_avg_{p_flip:.2f}.pdf", dpi=200,bbox_inches='tight')



    # plt.figure(figsize=(4.5,3))
    # plt.scatter(np.arange(4*L), eigvals, s=1)
    # plt.axhline(0, color='k', lw=.5)
    # plt.xlabel('index')
    # plt.ylabel('eigenvalue')
    # plt.title(f'Disordered Majorana ladder  (L={L}, p={p_flip}, seed={seed})')
    # plt.tight_layout()
    # plt.show()







if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)