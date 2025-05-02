import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def gidx(r, s):
    return 4*r + s

def build_H(L, flips01, Jx=1.0, Jy=1.0, Jz=1.0):
    """
    Only the (r,0)-(r,1) z-bonds use flips01[r]; the (r,2)-(r,3) bonds are always +1.
    """
    H = np.zeros((4*L, 4*L), dtype=np.complex128)
    for r in range(L):
        rp = (r+1) % L

        # only one family of z-rungs
        s01 = flips01[r]
        i,j = gidx(r,0), gidx(r,1)
        H[i,j] =  1j*s01*Jz
        H[j,i] = -1j*s01*Jz

        # the other z-rungs are fixed +1
        i,j = gidx(r,2), gidx(r,3)
        H[i,j] =  1j*Jz
        H[j,i] = -1j*Jz

        # intra-cell x,y
        i,j = gidx(r,0), gidx(r,2)
        H[i,j] =  1j*Jx;     H[j,i] = -1j*Jx
        i,j = gidx(r,3), gidx(r,1)
        H[i,j] =  1j*Jy;     H[j,i] = -1j*Jy

        # inter-cell x,y
        i,j = gidx(r,3),  gidx(rp,1)
        H[i,j] =  1j*Jy;     H[j,i] = -1j*Jy
        i,j = gidx(rp,0), gidx(r,2)
        H[i,j] =  1j*Jx;     H[j,i] = -1j*Jx

    return 0.5*H

def majorana_disordered_pair(L, p_flip, seed=None):
    """
    - flips01: Bernoulli(p_flip) on the rungs (r,0)-(r,1)
    - flips23: always +1, untouched
    - then flip exactly one random entry in flips01 to get the primed sector
    """
    rng = np.random.default_rng(seed)

    # only one family of z-bonds is random
    flips01 = np.where(rng.random(L) < p_flip, -1, +1)
    # pick one bond to flip
    b = rng.integers(0, L)
    flips01p = flips01.copy()
    flips01p[b] *= -1

    H_F  = build_H(L, flips01)
    H_Fp = build_H(L, flips01p)
    return H_F, H_Fp, b

def overlap_squares(H1, H2):
    E1, V1 = la.eigh(H1)
    E2, V2 = la.eigh(H2)
    ov2    = np.abs(np.sum(np.conj(V1)*V2, axis=0))**2
    return E1, E2, ov2

def main():
    L      = 200
    p_flip = 0.1
    seed   = 2025

    H_F, H_Fp, flipped = majorana_disordered_pair(L, p_flip, seed)
    E_F, E_Fp, ov_F     = overlap_squares(H_F, H_Fp)

    # zero-flux baseline: all +1, then flip the same bond
    zeros = np.ones(L, dtype=int)
    zeros_p = zeros.copy(); zeros_p[flipped] *= -1
    H0   = build_H(L, zeros)
    H0p  = build_H(L, zeros_p)
    E0, E0p, ov0 = overlap_squares(H0, H0p)

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(ov_F, 'o', label='Random flux', ms=4)
    plt.plot(ov0, '^', label='Zero-flux',  ms=4)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Band index $n$')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()