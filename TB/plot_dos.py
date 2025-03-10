import re
import math 
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


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


def main(total, cmdargs):

    # Read the data
    data = readfArray("DOS0_L3000.dat", Complex = False)

    fig, ax = plt.subplots(1, 1,  figsize=(8,6))  # 1 row 1 col

    ax.plot(np.arange(0, 0.61, 0.01), data, label=r"${\rm DOS}(\omega=0)$", color='blue', linewidth=2)
    
    
    ax.legend(loc='best', fontsize=18, frameon = False)
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
    # plt.minorticks_on()
    
    ax.set_xlabel(r"$F$", fontsize=18)
    ax.set_ylabel(r"${\rm DOS}(\omega=0)$", fontsize=18)
    # plt.show()
    plt.savefig("figure.pdf", dpi=600, bbox_inches='tight')



if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
