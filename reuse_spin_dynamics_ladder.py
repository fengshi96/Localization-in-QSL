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

def measure_spin_densities(psi, lattice, model_params, site='all'):
    """
    Measure spin density operators for a given psi.
    
    Inputs:
    psi: any state, e.g. a time-dependent state exp(-iHt)|perturbed> modified in-place in mps environment
    lattice: Attribute lattice of the model, e.g. Honeycomb.lat
    site: list of sites in which available energy density operators are matched in terms of operators' site of reference

    Output:
    expectation value <psi| op |psi>

    The lattice is labeled as follows (e.g. for a Lx = 8 ladder under OBC):
    1 - x - 3 - y - 5 - x - 7 - y - 9 - x - 11 - y - 13 - x - 15
    |       |       |       |       |       |        |        |
    z       z       z       z       z       z        z        z
    |       |       |       |       |       |        |        |
    0 - y - 2 - x - 4 - y - 6 - x - 8 - y - 10 - x - 12 - y - 14
    
    or the folded label, which works decently under PBC (longest range connection is fixed at 4 sites):
    - y - 1 - x - 5 - y - 9 - x - 13 - y - 15 - x - 11 - y - 7 - x - 3 - y -
          |       |       |       |        |        |        |       |
          z       z       z       z        z        z        z       z
          |       |       |       |        |        |        |       |
    - x - 0 - y - 4 - x - 8 - y - 12 - x - 14 - y - 10 - x - 6 - y - 2 - x -
    """
    if site == 'all':
        site = range(lattice.N_sites)
    elif type(site) == int:
        # if a single site as int, turn it into a list e.g. 1 -> [1]
        s_list = [site, ]
        site = s_list

    # one-point terms x+y+z for A and B sublattices
    ops_Ax = []
    ops_Ay = []
    ops_Az = []
    ops_Bx = []
    ops_By = []
    ops_Bz = []

    ops_lAx = []
    ops_lAy = []
    ops_lAz = []
    ops_lBx = []
    ops_lBy = []
    ops_lBz = []

    ops_rAx = []
    ops_rAy = []
    ops_rAz = []
    ops_rBx = []
    ops_rBy = []
    ops_rBz = []

    exp_values = []

    # energy xx + yy + zz
    # We creat a mask that incompasses the largest support \Union(op_a, op_b)
    # such that the list of op_a matches that of op_b
    #                  0               1                2                3             4                5
    op_mask = [("Id", [-1], 0), ("Id", [-1], 1), ('Id', [0], 0), ('Id', [0], 1), ("Id", [1], 0), ("Id", [1], 1)]
    mps_inds = lattice.possible_multi_couplings(op_mask)[0]

    op_found = False
    for inds in mps_inds:
        if inds[0] in site:
            print(inds[0], "is in the site list")
            # use inds[0] as the site of reference for an operator
            op_found = True

            # x+y+z
            op_Ax = [('Sigmax', inds[2])]
            op_Ay = [('Sigmay', inds[2])]
            op_Az = [('Sigmaz', inds[2])]

            op_Bx = [('Sigmax', inds[3])]
            op_By = [('Sigmay', inds[3])]
            op_Bz = [('Sigmaz', inds[3])]

            op_lAx = [('Sx', inds[0])]
            op_lAy = [('Sy', inds[0])]
            op_lAz = [('Sz', inds[0])]

            op_lBx = [('Sx', inds[1])]
            op_lBy = [('Sy', inds[1])]
            op_lBz = [('Sz', inds[1])]

            op_rAx = [('Sx', inds[4])]
            op_rAy = [('Sy', inds[4])]
            op_rAz = [('Sz', inds[4])]

            op_rBx = [('Sx', inds[5])]
            op_rBy = [('Sy', inds[5])]
            op_rBz = [('Sz', inds[5])]

            ops_Ax.append(op_Ax)
            ops_Ay.append(op_Ay)
            ops_Az.append(op_Az)
            ops_Bx.append(op_Bx)
            ops_By.append(op_By)
            ops_Bz.append(op_Bz)

            ops_lAx.append(op_lAx)
            ops_lAy.append(op_lAy)
            ops_lAz.append(op_lAz)
            ops_lBx.append(op_lBx)
            ops_lBy.append(op_lBy)
            ops_lBz.append(op_lBz)
            
            ops_rAx.append(op_rAx)
            ops_rAy.append(op_rAy)
            ops_rAz.append(op_rAz)
            ops_rBx.append(op_rBx)
            ops_rBy.append(op_rBy)
            ops_rBz.append(op_rBz)

            print("finished appending operators")

    if not op_found:
        raise ValueError(site, "No available energy operator found according to the list of sites!")
    


    hx = 1
    hy = 1
    hz = 1
    print("-----------\n","Measuring the following spin density operators:")
    for op_Ax, op_Ay, op_Az, op_Bx, op_By, op_Bz, \
         op_lAx, op_lAy, op_lAz, op_lBx, op_lBy, op_lBz, op_rAx, op_rAy, op_rAz, op_rBx, op_rBy, op_rBz \
            in zip(ops_Ax, ops_Ay, ops_Az, ops_Bx, ops_By, ops_Bz, \
                   ops_lAx, ops_lAy, ops_lAz, ops_lBx, ops_lBy, ops_lBz, ops_rAx, ops_rAy, ops_rAz, ops_rBx, ops_rBy, ops_rBz):
        
        print(
            f"+{hx:.2f}", op_Ax, f"+{hy:.2f}", op_Ay, f"+{hz:.2f}", op_Az, \
            f"+{hx:.2f}", op_Bx, f"+{hy:.2f}", op_By, f"+{hz:.2f}", op_Bz, \
            f"+{hx:.2f}", op_lAx, f"+{hy:.2f}", op_lAy, f"+{hz:.2f}", op_lAz, \
            f"+{hx:.2f}", op_lBx, f"+{hy:.2f}", op_lBy, f"+{hz:.2f}", op_lBz, \
            f"+{hx:.2f}", op_rAx, f"+{hy:.2f}", op_rAy, f"+{hz:.2f}", op_rAz, \
            f"+{hx:.2f}", op_rBx, f"+{hy:.2f}", op_rBy, f"+{hz:.2f}", op_rBz)
        print("-----------\n")
        
        expvalue =  hx * psi.expectation_value_term(op_Ax) + hy * psi.expectation_value_term(op_Ay) + hz * psi.expectation_value_term(op_Az) \
                    + hx * psi.expectation_value_term(op_Bx) + hy * psi.expectation_value_term(op_By) + hz * psi.expectation_value_term(op_Bz) \
                    + hx * psi.expectation_value_term(op_lAx) + hy * psi.expectation_value_term(op_lAy) + hz * psi.expectation_value_term(op_lAz) \
                    + hx * psi.expectation_value_term(op_lBx) + hy * psi.expectation_value_term(op_lBy) + hz * psi.expectation_value_term(op_lBz) \
                    + hx * psi.expectation_value_term(op_rAx) + hy * psi.expectation_value_term(op_rAy) + hz * psi.expectation_value_term(op_rAz) \
                    + hx * psi.expectation_value_term(op_rBx) + hy * psi.expectation_value_term(op_rBy) + hz * psi.expectation_value_term(op_rBz)
        exp_values.append(expvalue) 
    

    return exp_values





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
    M = Kitaev_Ladder(model_params)
    site = range(0, M.lat.N_sites, 4)


    timeCuts = []
    i = 1
    for num, fname in files_with_numbers:
        print(f"Processing t_step = {num} using file {fname}")
        with h5py.File(fname, 'r') as f:
            info = hdf5_io.load_from_hdf5(f)

        evolved_time = info['evolved_time']
        psi = info['last MPS']
        result = measure_spin_densities(psi, M.lat, model_params, site=site)
        timeCuts.append(result)


        print(f"Overlap: {result}")

        P_returns[i, 0] = evolved_time
        P_returns[i, 1] = p_return
        i += 1

        





    
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
