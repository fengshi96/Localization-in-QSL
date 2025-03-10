import numpy as np
from model_ladder import Kitaev_Ladder
import os, re
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


def main(**kwargs):
    # Directory containing the files
    directory = "energy_time_states/"

    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter files that start with a digit and extract the number
    files_with_numbers = []
    for filename in all_files:
        # Use regex to check if filename starts with digits and extract them
        match = re.match(r"(\d+)", filename)
        if match:
            number = int(match.group(1))
            filename = directory + filename
            files_with_numbers.append((number, filename))

    # Sort the list by the extracted number (first element of the tuple)
    files_with_numbers.sort(key=lambda x: x[0])

    # Optionally, print the sorted list
    for num, fname in files_with_numbers:
        print(num, fname)

    gs_file = kwargs['gs_file']  # file containing the ground state and model params
    with h5py.File(gs_file, 'r') as f:
        gs_info = hdf5_io.load_from_hdf5(f)

    model_params = gs_info['model_params']
    phi = gs_info['gs']


    # prepare bra <0|O for <|O exp(-iHt) O|> = <phi|psi>, where phi = <0|O and psi = exp(-iHt) O|gs>
    for j in np.array([model_params['Lx'] - 1, model_params['Lx']]):
        print("Perturbing site ", j)
        op_type = kwargs['op_type']
        phi.apply_local_op(j, op_type, unitary=False, renormalize=False)


    P_returns = np.zeros((len(files_with_numbers)+1, 2))
    P_returns[0, 0] = 0
    P_returns[0, 1] = np.abs(phi.overlap(phi))**2

    i = 1
    for num, fname in files_with_numbers:
        print(f"Processing t_step = {num} using file {fname}")
        with h5py.File(fname, 'r') as f:
            info = hdf5_io.load_from_hdf5(f)

        evolved_time = info['evolved_time']
        psi = info['last MPS']
        overlap = phi.overlap(psi)
        p_return = np.abs(overlap)**2

        print(f"Overlap: {overlap}")
        print(f"p_return: {p_return}")

        P_returns[i, 0] = evolved_time
        P_returns[i, 1] = p_return
        i += 1

    print(P_returns)
    printfArray(P_returns, "return_prob_overlap.txt")
        





    
# ----------------------------------------------------------
# ----------------- Run main() ----------------
# ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.argv ## get the input argument
    total = len(sys.argv)

    op_type = 'Sigmaz'
    gs_file = "./ground_states/GS_L10defaultchi400_K-1.00Fx-0.00Fy-0.00Fz-0.00.h5"

    main(gs_file=gs_file, op_type=op_type)