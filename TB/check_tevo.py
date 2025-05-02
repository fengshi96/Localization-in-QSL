import sys
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def gidx(r, s):
    return 4*r + s

def build_H(L, flips01, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Build 4L×4L Majorana H with only the (r,0)-(r,1) z-bonds flipped by flips01[r].
    The (r,2)-(r,3) bonds are fixed +1.
    """
    H = np.zeros((4*L, 4*L), dtype=np.complex128)
    for r in range(L):
        rp = (r+1)%L
        # lower rung
        s01 = flips01[r]
        i,j = gidx(r,0), gidx(r,1)
        H[i,j] =  1j*s01*Jz;  H[j,i] = -1j*s01*Jz
        # upper rung fixed
        i,j = gidx(r,2), gidx(r,3)
        H[i,j] =  1j*Jz;     H[j,i] = -1j*Jz
        # intra‐cell x,y
        i,j = gidx(r,0), gidx(r,2)
        H[i,j] =  1j*Jx;     H[j,i] = -1j*Jx
        i,j = gidx(r,3), gidx(r,1)
        H[i,j] =  1j*Jy;     H[j,i] = -1j*Jy
        # inter‐cell x,y
        i,j = gidx(r,3),  gidx(rp,1)
        H[i,j] =  1j*Jy;     H[j,i] = -1j*Jy
        i,j = gidx(rp,0), gidx(r,2)
        H[i,j] =  1j*Jx;     H[j,i] = -1j*Jx
    return 0.5*H


def majorana_pair(L, p_flip, seed=None):
    """
    Returns H_F, H_Fp, and the bond‐index b flipped:
      - flips01[r] ~ Bernoulli(p_flip) on lower rungs
      - flips01p flips exactly one randomly chosen bond b
    """
    rng = np.random.default_rng(seed)
    flips01 = np.where(rng.random(L) < p_flip, -1, +1)
    b = rng.integers(L)
    flips01p = flips01.copy(); flips01p[b] *= -1
    return build_H(L, flips01), build_H(L, flips01p), b





def majorana_pair_assigned(L, p_flip, flip_index, seed=None):
    """
    Build a Majorana‐pair (H_F, H_F') with exactly one prescribed bond flip.

    Parameters
    ----------
    L : int
      number of unit cells
    p_flip : float
      probability for each lower‐leg z‐bond to be flipped in the base sector F
    flip_index : int
      which lower‐leg z‐bond (0 <= flip_index < L) to flip when constructing F'
    seed : int or None
      RNG seed for reproducibility

    Returns
    -------
    H_F  : ndarray (4L×4L)
      Majorana Hamiltonian in the random‐flux sector F
    H_Fp : ndarray (4L×4L)
      Majorana Hamiltonian in the sector F′ with bond #flip_index flipped
    flip_index : int
      the same index you passed in, for bookkeeping
    """
    rng = np.random.default_rng(seed)

    # 1) initial random flux on the lower z‐rungs only
    flips01 = np.where(rng.random(L) < p_flip, -1, +1)

    # 2) make a copy and flip exactly the prescribed one
    if not (0 <= flip_index < L):
        raise ValueError(f"flip_index must be in [0, {L}), got {flip_index}")
    flips01p = flips01.copy()
    flips01p[flip_index] *= -1

    # 3) build Hamiltonians
    H_F  = build_H(L, flips01)   # as before
    H_Fp = build_H(L, flips01p)

    return H_F, H_Fp, flip_index



def transition_probabilities(H, i, j, t_vals):
    """
    Compute P(t) = |<i|e^{-i H t}|j>|^2 for t in t_vals
    via spectral decomposition.
    """
    E, V = la.eigh(H)
    # matrix elements V[i,n] * conj(V[j,n])
    A = V[i,:] * np.conjugate(V[j,:])   # shape (4L,)
    # for each t, amplitude = sum_n A[n]*exp(-i E[n] t)
    # vectorize: compute outer product E * t_vals
    phi = np.exp(-1j * np.outer(E, t_vals))  # shape (4L, len(t_vals))
    amps = A[:,None] * phi                    # broadcast → (4L, nt)
    ujt = np.sum(amps, axis=0)                # shape (nt,)
    return np.abs(ujt)**2

def main():
    L       = 600
    p_flip  = 0.5
    seed    = 2025
    i_site  = gidx(0,0)   # choose local Majorana modes by index
    j_site  = gidx(100,0)
    t_max   = 800.0
    nt      = 120


    # time grid
    t_vals = np.linspace(0, t_max, nt)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, sharey=False)

    for ax, p_flip, title, ylims in zip(
        axes,
        [0.0, 0.5],
        [r"(a) Uniform $\pi$ flux ($p=0.0$)", r"(b) Random flux ($p=0.5$)"],
        [(-1e-3, 2.2e-2), (-1e-7, 3e-6)]      # example y‐limits for each panel
    ):
        # build the two Hamiltonians
        H_F, H_Fp, _ = majorana_pair_assigned(L, p_flip, flip_index=j_site // 4, seed=seed)

        # compute P(t) for each
        P_F  = transition_probabilities(H_F,  i_site, j_site, t_vals)
        P_Fp = transition_probabilities(H_Fp, i_site, j_site, t_vals)

        # plot
        ax.plot(t_vals, P_F,  '-', lw=2, label='Original F', color='red')
        ax.plot(t_vals, P_Fp, '-', lw=1, label="F with one bond flipped", color='blue')

        # scientific y‐axis
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # set the per‐panel ylim
        ax.set_ylim(ylims)
        ax.set_xlim(xmin=0, xmax=t_max)

        # styling
        ax.set_ylabel(r'$|\langle i|U_F(t)|j\rangle|^2$', fontsize=18)
        ax.set_title(title, loc='left', pad=0, x=0.02, y=0.9, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)

        # one‐row, two‐col legend
        ax.legend(ncol=1, frameon=False, loc='upper right', fontsize=13)

    # common x‐axis label
    axes[-1].set_xlabel(r'$t/J^{-1}$', fontsize=20)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./timeEvolCheck.pdf", dpi=200,bbox_inches='tight')




if __name__=='__main__':
    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    main()