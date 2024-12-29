import re
import math 
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import colormaps
import numpy as np

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')

    # main codes
    Spins = readfArray('Spins_h0.60.dat')
#     Spins = Spins - Spins[0, :]
    Spins = -Spins # np.abs(Spins)
#     print(np.max(Spins))

    fig, ax = plt.subplots(1, 1,  figsize=(8,6))  # 1 row 1 col

    vmax = 0.1
    vmin = -0.4
    colmap = colormaps['magma']
#     colmap = colormaps['seismic']
    ax.imshow(Spins, cmap=colmap, origin='lower', aspect=0.006, vmin=vmin, vmax=vmax)
    
#     ax.legend(loc='best', fontsize=18, frameon = False)
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
    # plt.minorticks_on()

    ax.text(0.6, 1.07, "(a) Je, h=0.6", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
    
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$t$", fontsize=18)
    # plt.show()
    plt.savefig("figure.pdf", dpi=300, bbox_inches='tight')
    




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
