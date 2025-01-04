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
        exp_values.append(expvalue) 
    

    return exp_values




def measure_fluxes(psi, lattice, site='all'):
    """
    Measure flux operators for a given psi.
    For a ladder, there are three types of fluxes: Ws1 (square faces a), Ws2 (square faces b) and Wp (hexagons).
    
    Inputs:
    psi: any state, e.g. a time-dependent state exp(-iHt)|perturbed> modified in-place in mps environment
    lattice: Attribute lattice of the model, e.g. Honeycomb.lat
    site: list of sites in which available energy density operators are matched in terms of operators' site of reference

    Output:
    expectation value <psi| flux_op |psi>
    """
    if site == 'all':
        site = range(lattice.N_sites)
    elif type(site) == int:
        # if a single site as int, turn it into a list e.g. 1 -> [1]
        s_list = [site, ]
        site = s_list

    # two-point terms xx+yy+zz
    ops_s1 = []
    ops_s2 = []
    ops_wp = []

    exp_values_s1 = []
    exp_values_s2 = []
    exp_values_wp = []

    # We creat a mask that incompasses the largest support \Union(op_s1, op_s2) -> op_w
    #                  0               1                2                3             4                5
    op_mask = [("Id", [-1], 0), ("Id", [-1], 1), ('Id', [0], 0), ('Id', [0], 1), ("Id", [1], 0), ("Id", [1], 1)]
    mps_inds = lattice.possible_multi_couplings(op_mask)[0]

    op_found = False
    for inds in mps_inds:
        if inds[0] in site:
            # use inds[0] as the site of reference for an operator
            op_found = True

            # locations of the square faces s1 and s2 and hexagons
            op_s1 = [('Sigmax', inds[0]), ('Sigmay', inds[1]), ('Sigmax', inds[2]), ('Sigmay', inds[3])]
            op_s2 = [('Sigmay', inds[2]), ('Sigmax', inds[3]), ('Sigmay', inds[4]), ('Sigmax', inds[5])]
            op_wp = [('Sigmax', inds[0]), ('Sigmay', inds[1]), ('Sigmaz', inds[3]), ('Sigmax', inds[5]), ('Sigmay', inds[4]), ('Sigmaz', inds[2])]
            
            ops_s1.append(op_s1)
            ops_s2.append(op_s2)
            ops_wp.append(op_wp)

    if not op_found:
        raise ValueError(site, "No available energy operator found according to the list of sites!")
    
    print("-----------\n","Measuring the S1 operators:")
    for op_s1 in ops_s1:
        print(op_s1)
        expvalue = psi.expectation_value_term(op_s1) + 0.
        exp_values_s1.append(expvalue) 

    print("-----------\n","Measuring the S2 operators:")
    for op_s2 in ops_s2:
        print(op_s2)
        expvalue = psi.expectation_value_term(op_s2) + 0.
        exp_values_s2.append(expvalue) 

    print("-----------\n","Measuring the Wp operators:")
    for op_wp in ops_wp:
        print(op_wp)
        expvalue = psi.expectation_value_term(op_wp) + 0.
        exp_values_wp.append(expvalue) 
    
    return exp_values_s1, exp_values_s2, exp_values_wp



def measure(measurements, env, M, model_params, site):
    """ function to measure several observables
    Parameters:
    measurements: thing to be measured
    evn: class MPSEnvironment(gs, psi)
    """
    # Attribute MPSEnvironment.ket() is inherited from BaseEnvironment; expectation_value() from class BaseMPSExpectationValue; Default is for the entire MPS
    # see https://github.com/tenpy/tenpy/blob/main/tenpy/networks/mps.py#L5389


    measurements['energy'] = measure_energy_densities(env.ket, M.lat, model_params, site)
    measurements['Ws1'], measurements['Ws2'], measurements['Wp'] = measure_fluxes(env.ket, M.lat, site)


    env.clear()


def run_statics(**kwargs):
    """ run time evolution and measure hole correlation functions
    kwargs:
        gs_file: the name of the file that contains the ground state and model parameters
        op_type: operator type, e.g. Sz, Sx, Sy
        j_unit_cell: location j of the operator, offset from the center
        dt: time step
        t_method: TDVP or ExpMPO
        tsteps_init: time span using SVD compression for ExpMPO
        tsteps_cont: time span using variational compression for ExpMPO
        save_state: whether we want to save the full states at some time steps (Default = False)
        evolve_gs: whether we want to also time evolve the ground state (Default = False)
    """
    # load ground state
    gs_file = kwargs['gs_file']  # file containing the ground state and model params
    with h5py.File('./ground_states/'+gs_file, 'r') as f:
        state = hdf5_io.load_from_hdf5(f)

    model_params = state['model_params']
    hx = model_params['Fx']
    hy = model_params['Fy']
    hz = model_params['Fz']
    gs = state['gs']
    psi = gs.copy()

    M = Kitaev_Ladder(model_params)
    env = MPSEnvironment(gs, psi)
    chi_max = kwargs['chi_max']

    measurements = {}

    site = range(0, M.lat.N_sites, 4)
    if model_params['order'] == 'folded':
        site = []
        j = 0
        while j < M.lat.N_sites:
            site.append(j)
            j += 8
        
        j -= 10

        while j > 0:
            site.append(j)
            j -= 8

    print(site, "site")



    measure(measurements, env, M, model_params, site)
    print(measurements['energy'], "energy")
    print(measurements['Ws1'], "Ws1")
    print(measurements['Ws2'], "Ws2")
    print(measurements['Wp'], "Wp")

    
    statics = {"gs_file": gs_file, "fields": str(hx)+str(hy)+str(hz), 
                        "chi_max": chi_max, "model_params": model_params, "pos": M.positions(), 
                        "measurements": measurements}
    
    with h5py.File(f'gs_statics.h5', 'w') as f:
        hdf5_io.save_to_hdf5(f, statics)


# ----------------------------------------------------------
# ----------------- Debug run_time() ----------------
# ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.argv ## get the input argument
    total = len(sys.argv)
    if total !=2:
        raise("missing or having wrong arguments! 1 cmdargs gs_file!")
    cmdargs = sys.argv

    gs_file = cmdargs[1]
    chi_max = 80

    run_statics(gs_file=gs_file, chi_max=chi_max)