import numpy as np
import matplotlib.pyplot as plt
import sys

# ------------------------------------------------------------
#  gap-ratio utility
# ------------------------------------------------------------
def gap_ratio(energies):
    """
    Plot the gap-ratio distribution and return <r>.

    Parameters
    ----------
    energies : 1-D ndarray
        All many-body eigen-energies from exact diagonalisation.
    nbins : int
        Number of histogram bins.
    title : str
        Extra text for the plot title.

    Returns
    -------
    r_vals : ndarray
        The list of gap ratios r_n.
    r_mean : float
        The average gap ratio.
    """
    E = np.sort(energies)               # ensure ascending order
    delta = np.diff(E)                  # level spacings Δ_n
    delta = delta[delta != 0.0]
    r_vals = np.minimum(delta[:-1], delta[1:]) / np.maximum(delta[:-1], delta[1:])
    r_mean = r_vals.mean()

    return r_vals, r_mean


def P_pois(r):
    """Poisson gap‐ratio probability density P(r)=2/(1+r)^2."""
    return 2.0/(1.0 + r)**2

# r_pois_mean = 2*np.log(2) - 1



def main(total, cmdargs):
    """
    Main function to execute the gap ratio calculation.

    Parameters
    ----------
    total : int
        Total number of command line arguments.
    cmdargs : list
        List of command line arguments.
    """
    if total != 1:
        print("Usage: python level_stats.py <filename>")
        return

    filename = "energy_levels.dat"
    # Evals = readfArray(filename, Complex=False)
    Evals = np.loadtxt(filename, dtype=float)

    r_vals, r_mean = gap_ratio(Evals)
    print(r_vals)

    nbins = 20
    dens, bins = np.histogram(r_vals, bins=nbins, density=True)
    centers = 0.5*(bins[:-1] + bins[1:])   # mid-points of each bin

    plt.figure(figsize=(8,6))
    plt.scatter(centers, dens, s=50, color='C1', edgecolor='k')
    plt.plot(centers, P_pois(centers), 'r--', lw=2,
         label=r'Poisson fit $2/(1+r)^2$')
    plt.axvline(r_mean, color='r', lw=1.5,
                label=fr'$\langle r\rangle = {r_mean:.3f}$')

    plt.xlabel(r'gap ratio $r$')
    plt.ylabel('estimated density')
    plt.legend()
    plt.tight_layout()
    plt.show()






def readfArray(str, Complex = False):
    def toCplx(s):
        if "j" in s:
            return complex(s)
        else:
            repart = float(s.split(",")[0].split("(")[1])
            impart = float(s.split(",")[1].strip("j").split(")")[0])
            return complex(repart,impart)
    
    file = open(str,'r')
    lines = file.readlines()
    file.close()

    # Determine shape:
    row = len(lines)
    testcol = lines[0].strip("\n").rstrip().split()
    col = len(testcol)  # rstip to rm whitespace at the end 

    if Complex:
        m = np.zeros((row, col), dtype=complex)
        for i in range(row):
            if lines[i] != "\n":
                line = lines[i].strip("\n").rstrip().split()
                # print(line)
                for j in range(col):
                    val = toCplx(line[j])
                    m[i, j] = val
        
    else:
        m = np.zeros((row, col), dtype=float)
        for i in range(row):
            if lines[i] != "\n":
                line = lines[i].strip("\n").rstrip().split()
                # print(line)
                for j in range(col):
                    val = float(line[j])
                    m[i, j] = val
    return m


if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
