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
    if total != 2:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')

    # main codes
    h = float(cmdargs[1])
    name_Je = 'Je_h' + f'{h:.2f}' + '_long.dat'
    name_Js = 'Js_h' + f'{h:.2f}' + '_long.dat'
    Jet = readfArray(name_Je)
    Jst = readfArray(name_Js)[:, 0:7]
#     Jet = Jet - Jet[0, :]
#     Jst = Jst - Jst[0, :]
#     Jet = np.abs(Jet)
#     Jst = np.abs(Jst)

    fig, (ax0, ax1) = plt.subplots(1, 2,  figsize=(16,6))  # 1 row 1 col

    vmax = 0.1
    vmin = -0.1
    colmap = colormaps['seismic']
#     colmap = colormaps['seismic']
    ax0.imshow(Jet, cmap=colmap, origin='lower', aspect=0.005, vmin=vmin, vmax=vmax)
    ax1.imshow(Jst, cmap=colmap, origin='lower', aspect=0.005, vmin=vmin, vmax=vmax)
    
#     ax.legend(loc='best', fontsize=18, frameon = False)
    ax0.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
    ax1.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
    # plt.minorticks_on()

    ax0.text(0.6, 1.07, "(a) Je, h=" + str(h), transform=ax0.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
    ax1.text(0.6, 1.07, "(b) Js, h=" + str(h), transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
    
    ax0.set_xlabel(r"$x$", fontsize=18)
    ax1.set_xlabel(r"$x$", fontsize=18)
    ax0.set_ylabel(r"$t$", fontsize=18)
    # plt.show()
    fig_name = 'figure_h' + str(round(h, 2)) + '_long.pdf'
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    




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
