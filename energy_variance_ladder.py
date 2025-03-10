import numpy as np
from model_ladder import Kitaev_Ladder
import os
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

def measure_energy_densities(psi, lattice, model_params, site='all'):
    """
    Measure energy density operators for a given psi.
    
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

    # two-point terms xx+yy+zz
    ops_mz = []

    ops_lx = []
    ops_ly = []
    ops_lz = []

    ops_rx = []
    ops_ry = []
    ops_rz = []

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

            # xx+yy+zz
            op_lz = [('Sz', inds[0]), ('Sigmaz', inds[1])]  
            op_ly = [('Sigmay', inds[0]), ('Sigmay', inds[2])]  
            op_lx = [('Sigmax', inds[1]), ('Sigmax', inds[3])]  

            op_rz = [('Sz', inds[4]), ('Sigmaz', inds[5])]  
            op_ry = [('Sigmay', inds[3]), ('Sigmay', inds[5])]  
            op_rx = [('Sigmax', inds[2]), ('Sigmax', inds[4])]  

            op_mz = [('Sigmaz', inds[2]), ('Sigmaz', inds[3])]  

            ops_lz.append(op_lz)
            ops_ly.append(op_ly)
            ops_lx.append(op_lx)
            ops_rz.append(op_rz)
            ops_ry.append(op_ry)
            ops_rx.append(op_rx)
            ops_mz.append(op_mz)


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
    


    hx = model_params['Fx']
    hy = model_params['Fy']
    hz = model_params['Fz']
    print("-----------\n","Measuring the following energy density operators:")
    for op_lz, op_lx, op_ly, op_rz, op_rx, op_ry, op_mz, op_Ax, op_Ay, op_Az, op_Bx, op_By, op_Bz, \
         op_lAx, op_lAy, op_lAz, op_lBx, op_lBy, op_lBz, op_rAx, op_rAy, op_rAz, op_rBx, op_rBy, op_rBz \
            in zip(ops_lz, ops_lx, ops_ly, ops_rz, ops_rx, ops_ry, ops_mz, ops_Ax, ops_Ay, ops_Az, ops_Bx, ops_By, ops_Bz, \
                   ops_lAx, ops_lAy, ops_lAz, ops_lBx, ops_lBy, ops_lBz, ops_rAx, ops_rAy, ops_rAz, ops_rBx, ops_rBy, ops_rBz):
        
        print(op_lz, " + ", op_lx, "+", op_ly, "+", op_rz, "+", op_rx, "+", op_ry, "+", op_mz, \
            f"+{hx:.2f}", op_Ax, f"+{hy:.2f}", op_Ay, f"+{hz:.2f}", op_Az, \
            f"+{hx:.2f}", op_Bx, f"+{hy:.2f}", op_By, f"+{hz:.2f}", op_Bz, \
            f"+{hx:.2f}", op_lAx, f"+{hy:.2f}", op_lAy, f"+{hz:.2f}", op_lAz, \
            f"+{hx:.2f}", op_lBx, f"+{hy:.2f}", op_lBy, f"+{hz:.2f}", op_lBz, \
            f"+{hx:.2f}", op_rAx, f"+{hy:.2f}", op_rAy, f"+{hz:.2f}", op_rAz, \
            f"+{hx:.2f}", op_rBx, f"+{hy:.2f}", op_rBy, f"+{hz:.2f}", op_rBz)
        print("-----------\n")
        
        expvalue = psi.expectation_value_term(op_lz) + psi.expectation_value_term(op_lx) + psi.expectation_value_term(op_ly) \
                    + psi.expectation_value_term(op_rz) + psi.expectation_value_term(op_rx) + psi.expectation_value_term(op_ry) + psi.expectation_value_term(op_mz)\
                    + hx * psi.expectation_value_term(op_Ax) + hy * psi.expectation_value_term(op_Ay) + hz * psi.expectation_value_term(op_Az) \
                    + hx * psi.expectation_value_term(op_Bx) + hy * psi.expectation_value_term(op_By) + hz * psi.expectation_value_term(op_Bz) \
                    + hx * psi.expectation_value_term(op_lAx) + hy * psi.expectation_value_term(op_lAy) + hz * psi.expectation_value_term(op_lAz) \
                    + hx * psi.expectation_value_term(op_lBx) + hy * psi.expectation_value_term(op_lBy) + hz * psi.expectation_value_term(op_lBz) \
                    + hx * psi.expectation_value_term(op_rAx) + hy * psi.expectation_value_term(op_rAy) + hz * psi.expectation_value_term(op_rAz) \
                    + hx * psi.expectation_value_term(op_rBx) + hy * psi.expectation_value_term(op_rBy) + hz * psi.expectation_value_term(op_rBz)

        # expvalue =  hx * psi.expectation_value_term(op_Ax)
        exp_values.append(expvalue) 
    

    return exp_values



def measure_energy_density_squared(psi, lattice, model_params, site=0):
    """
    Compute the expectation value <phi|h2^2|phi> using two sequential applications of h2.
    
    Parameters:
        psi (MPS): The initial state |phi> (an MPS object from TenPy).
        h2_ops (list): A list of tuples (j, op) where j is the site index and op is the local operator.
                       This list defines the local Hamiltonian density h2.
    
    Returns:
        energy_sq (float): The expectation value of h2^2.

    The lattice is labeled as follows (e.g. for a Lx = 10 ladder under OBC):
    1 - x - 3 - y - 5 - x - 7 - y - 9 - x - 11 - y - 13 - x - 15 - y - 17 - x - 19 - y - 21
    |       |       |       |       |       |        |        |        |        |        |
    z       z       z       z       z       z        z        z        z        z        z
    |       |       |       |       |       |        |        |        |        |        |
    0 - y - 2 - x - 4 - y - 6 - x - 8 - y - 10 - x - 12 - y - 14 - x - 16 - y - 18 - x - 20
    """

    if type(site) == int:
        # if a single site as int, turn it into a list e.g. 1 -> [1]
        s_list = [site, ]
        site = s_list
    else:
        raise ValueError("site must be an integer for h^2_i operator")


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

            # xx+yy+zz
            op_lz = [('Sz', inds[0]), ('Sigmaz', inds[1])]  
            op_ly = [('Sigmay', inds[0]), ('Sigmay', inds[2])]  
            op_lx = [('Sigmax', inds[1]), ('Sigmax', inds[3])]  

            op_rz = [('Sz', inds[4]), ('Sigmaz', inds[5])]  
            op_ry = [('Sigmay', inds[3]), ('Sigmay', inds[5])]  
            op_rx = [('Sigmax', inds[2]), ('Sigmax', inds[4])]  

            op_mz = [('Sigmaz', inds[2]), ('Sigmaz', inds[3])]  

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

    if not op_found:
        raise ValueError(site, "No available energy operator found according to the list of sites!")
    


    hx = model_params['Fx']
    hy = model_params['Fy']
    hz = model_params['Fz']


    # Make a copy of the original state |phi>
    psi_tmp = psi.copy()
    psi_orig = psi.copy()  # Ensure that this creates an independent copy of the state
    
    # First application: compute |psi> = h2 |phi>
    print("First application: compute |psi> = h2 |phi> ...")
    print("Applying on-site operators:")
    op, s = op_Ax[0]; print(op, s)
    psi.apply_local_op(s, hx * psi.sites[s].get_op(op), unitary=False, renormalize=False)

    op, s = op_Ay[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_Az[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_Bx[0]; print(op, s)
    psi_tmp.apply_local_op(s, hx * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_By[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_Bz[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_lAx[0]; print(op, s)
    psi_tmp.apply_local_op(s, hx * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_lAy[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_lAz[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_lBx[0]; print(op, s)
    psi_tmp.apply_local_op(s, hx * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_lBy[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_lBz[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_rAx[0]; print(op, s)
    psi_tmp.apply_local_op(s, hx * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_rAy[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_rAz[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_rBx[0]; print(op, s)
    psi_tmp.apply_local_op(s, hx * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_rBy[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()

    op, s = op_rBz[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()


    print("\nfinished applying on-site operators; now applying two-site operators:")
    (op1, s1), (op2, s2) = op_lz
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_ly
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_lx
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_rz
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_ry
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_rx
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_mz
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi.copy()  # note that psi_tmp is now h2 |phi>
    psi_orig2 = psi.copy()
    print(op1, s1); print(op2, s2, '\n')
        
    
    energy = psi_orig.overlap(psi)
    
    # # Second application: act with h2 again to obtain h2^2 |phi>
    print("Second application: compute |psi'> = h2 |psi> = h2 (h2 |phi>) ...")
    print("Applying on-site operators:")
    op, s = op_Ax[0]; print(op, s)
    psi.apply_local_op(s, hx * psi.sites[s].get_op(op), unitary=False, renormalize=False)


    op, s = op_Ay[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_Az[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_Bx[0]; print(op, s)
    psi_tmp.apply_local_op(s, hx * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_By[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_Bz[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_lAx[0]; print(op, s)
    psi_tmp.apply_local_op(s, hx * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_lAy[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_lAz[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_lBx[0]; print(op, s)
    psi_tmp.apply_local_op(s, hx * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_lBy[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_lBz[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_rAx[0]; print(op, s)
    psi_tmp.apply_local_op(s, hx * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_rAy[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_rAz[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_rBx[0]; print(op, s)
    psi_tmp.apply_local_op(s, hx * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_rBy[0]; print(op, s)
    psi_tmp.apply_local_op(s, hy * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()

    op, s = op_rBz[0]; print(op, s)
    psi_tmp.apply_local_op(s, hz * psi_tmp.sites[s].get_op(op), unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()


    print("\nfinished applying on-site operators; now applying two-site operators for the 2nd time:")
    (op1, s1), (op2, s2) = op_lz
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_ly
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_lx
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_rz
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_ry
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_rx
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()
    print(op1, s1); print(op2, s2, '\n')

    (op1, s1), (op2, s2) = op_mz
    psi_tmp.apply_local_op(s1, op1, unitary=False, renormalize=False)
    psi_tmp.apply_local_op(s2, op2, unitary=False, renormalize=False)
    psi = psi.add(psi_tmp, 1, 1)
    psi_tmp = psi_orig2.copy()
    print(op1, s1); print(op2, s2, '\n')
    
    # Now psi holds h2^2 |phi>.
    # Compute the overlap <phi_orig | (h2^2 |phi>) which gives the expectation value <phi|h2^2|phi>.
    energy_sq = psi_orig.overlap(psi)
    
    return energy, energy_sq




def main(**kwargs):
    # Directory containing the files
    directory = "./ground_states/"


    gs_file = kwargs['gs_file']  # file containing the ground state and model params
    with h5py.File(gs_file, 'r') as f:
        gs_info = hdf5_io.load_from_hdf5(f)

    model_params = gs_info['model_params']
    psi = gs_info['gs']


    j = model_params['Lx'] - 1
    print("Perturbing site ", j)
    op_type = kwargs['op_type']
    psi.apply_local_op(j, op_type, unitary=False, renormalize=False)

    psi2 = psi.copy()

    M = Kitaev_Ladder(model_params)
    energy, energy_sq = measure_energy_density_squared(psi, M.lat, model_params, site=j-2)
    print(energy)
    print(energy ** 2, energy_sq)


    energy_exp = measure_energy_densities(psi2, M.lat, model_params, site=j-2)  # j-2
    print(energy_exp)

        





    
# ----------------------------------------------------------
# ----------------- Run main() ----------------
# ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.argv ## get the input argument
    total = len(sys.argv)

    op_type = 'Sigmax'
    gs_file = "./ground_states/GS_L11defaultchi90_K-1.00Fx-0.70Fy-0.70Fz-0.70.h5"

    main(gs_file=gs_file, op_type=op_type)