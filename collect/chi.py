import re
import math 
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')

    # main codes
    data = readfArray("energies.dat")
    H0 = data[:, 0]
    E0 = data[:, 1]
    H1, E1 = derivative(E0, H0)
    H2, E2 = derivative(E1, H1)

    fig, ax = plt.subplots(1, 1,  figsize=(8,6))  # 1 row 1 col
    
    ax.plot(H2, E2)
    
#     ax.legend(loc='best', fontsize=18, frameon = False)
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
    # plt.minorticks_on()
    
    ax.set_xlabel(r"$h$", fontsize=18)
    ax.set_ylabel(r"$\chi$", fontsize=18)
#     ax.set_xlim(xmin=0.2)
#     ax.set_ylim(ymin=-150, ymax=0)
    # plt.show()
    plt.savefig("phase.pdf", dpi=300, bbox_inches='tight')



def derivative(Ys, Xs):
    derivatives = []
    Xnew = []
    for i in range(len(Xs) - 1):
        deltaY = Ys[i+1] - Ys[i]
        deltaX = Xs[i+1] - Xs[i]
        derivatives.append(deltaY / deltaX)
        Xnew.append(Xs[i] + deltaX / 2.0) 
    return np.array(Xnew), np.array(derivatives)


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
