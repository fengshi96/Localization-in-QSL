import re
import h5py
from tenpy.tools import hdf5_io
import math 
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import colormaps
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')

    # main codes
#     stat_file = 'gs_statics_L21x2_h0.00.h5'
#     with h5py.File(stat_file, 'r') as f:
#         stats = hdf5_io.load_from_hdf5(f)
#     gs_energies = stats['measurements']['energy']
#     print(gs_energies)


    Engs = readfArray('Engs_h0.76_21x2.dat')
    Engs_csl = readfArray('Engs_h0.12_21x2.dat')
#     Engs = Engs - gs_energies
#     Engs = Engs[0:600, :]
#     Engs = np.abs(Engs)
#     print(np.max(Engs))
    Total = sum(Engs[0, :])
    Total_csl = sum(Engs_csl[0, :])

    Engs = Engs - Total/(21*2*2)
    Engs_csl = Engs_csl - Total_csl/(21*2*2)

#     fig, (ax, ax1, ax2) = plt.subplots(1, 3,  figsize=(20,6))  # 1 row 1 col
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1], figure=fig)
    gs.update(wspace=0.3, hspace=0.3)  # Adjust spacing
    ax = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    vmax = 3.00
    vmin = -0.0
    colmap = colormaps['magma']
#     colmap = colormaps['seismic']
    img = ax.imshow(Engs, cmap=colmap, origin='lower', aspect=0.018, vmin=vmin, vmax=vmax)
    pos_a = ax.get_position()
    left_col_x0 = pos_a.x0  # Left position of the first column
    left_col_width = pos_a.width  # Width of the first column
    top_of_a = pos_a.y1  # Top of panel (a)

    # Add a horizontal color bar above panel (a), matching the first column width
    cbar_axes = fig.add_axes([left_col_x0, top_of_a + 0.045, left_col_width, 0.03])  # Slightly above panel (a)
    colorbar = fig.colorbar(img, cax=cbar_axes, orientation='horizontal')
#     colorbar.set_label("Energy Levels")
    colorbar.ax.tick_params(labelsize=12)

    cut_step = 100
    colormap = cm.get_cmap('BuPu')
    ncuts = 26
    colors = [colormap(i/ncuts) for i in range(2,2+ncuts)]  # 23, 25
    for i in range(0, ncuts*100, cut_step):# 2200
        t = i*0.01
        print(i, int(i/cut_step), t)
        print(len(colors))

        # only for coarse
#         print(Total, sum(Engs[i, :]), total/sum(Engs[i, :]))
        ratio = Total / sum(Engs[i, :])
        ax1.plot(Engs[i,:] * ratio, marker='s', color=colors[int(i/cut_step)], label=r"$t=$"+str(t))

#         ax1.plot(Engs[i,:], marker='s', color=colors[int(t)], label=r"$t=$"+str(t))
        ax.axhline(y=i, color = colors[int(i/cut_step)], linestyle='--', lw=0.5, alpha=0.6)
    
    ax.set_yticks(range(0, ncuts*100, 500))
    ax.set_yticklabels([int(i) for i in np.arange(0, ncuts*100, 500) * 0.01])
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
    ax1.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
    ax2.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
    ax2.set_xticks(range(0, ncuts*100, 500))
    ax2.set_xticklabels([int(i) for i in np.arange(0, ncuts*100, 500) * 0.01])
    # plt.minorticks_on()

    norm = mcolors.Normalize(vmin=0, vmax=ncuts)  # Normalize data range
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)  # Map colormap to the normalized range
    sm.set_array([])
    cbar_axes = fig.add_axes([0.46, 0.82, 0.2, 0.03])  # [left, bottom, width, height]
    colorbar = fig.colorbar(sm, cax=cbar_axes, orientation='horizontal')
    colorbar.set_label("t", fontsize=18)

#     ax.text(0.6, 1.07, r"(a) $\varepsilon(x, t)$", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
    ax1.text(3.5, 0.97, r"(b)", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
    ax2.text(3.5, 0.42, r"(c)", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='right')
    
    ax2.plot(Engs[:, 10], label='IGP')
    ax2.plot(Engs_csl[:, 10], label='CSL')
    ax2.legend(ncol=2, loc='upper center', fontsize=14, frameon = False)
#     ax2.axhline(y=Total/(21*2*2), color = 'blue', linestyle='--', lw=1)
#     ax2.axhline(y=Total_csl/(21*2*2), color = 'orange', linestyle='--', lw=1)

    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$t$", fontsize=18)
    ax1.set_xlabel(r"$x$", fontsize=18)
    ax1.set_ylabel(r"$\varepsilon(x,\{t\})$", fontsize=18)
    ax2.set_xlabel(r"$t$", fontsize=18)
    ax2.set_ylabel(r"$\varepsilon(x=10,t)$", fontsize=18)
    ax2.set_ylim(ymin=0.00, ymax=3)
#     ax2.set_xscale('log')
#     ax2.set_yscale('log')

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
