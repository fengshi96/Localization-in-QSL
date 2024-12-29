import numpy as np
# from model import Kitaev_Extended
import os
import sys
import h5py
# import pickle
import time
from tenpy.tools import hdf5_io
from tenpy.networks.mps import MPS, MPSEnvironment
from tenpy.networks.mpo import MPOEnvironment
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution
# from tenpy.algorithms.tdvp import TwoSiteTDVPEngine, SingleSiteTDVPEngine
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



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

stat_file = '../gs_statics.h5'
with h5py.File(stat_file, 'r') as f:
    stats = hdf5_io.load_from_hdf5(f)
gs_energies = stats['measurements']['energy']

sys.argv ## get the input argument
total = len(sys.argv)
if total !=2:
    raise("missing arguments! 1 cmdargs!")
cmdargs = sys.argv
name = cmdargs[1] 

with h5py.File(name, 'r') as f:
    model_params = hdf5_io.load_from_hdf5(f['model_params'])
    measurements = hdf5_io.load_from_hdf5(f['measurements'])
    time_keys = list(measurements.keys())


Eng = np.zeros((len(time_keys), model_params['Lx'] - 1), dtype=float)
for t_indx, time in enumerate(time_keys):
    Eng[t_indx, :] = np.array((measurements[time]['energy'])) - gs_energy


printfArray(Eng, 'Engs.dat')