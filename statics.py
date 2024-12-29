import numpy as np
from model import Kitaev_Extended
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
    expectation value <psi| cur_op |psi>
    """
    if site == 'all':
        site = range(lattice.N_sites)
    elif type(site) == int:
        # if a single site as int, turn it into a list e.g. 1 -> [1]
        s_list = [site, ]
        site = s_list

    # two-point terms xx+yy+zz
    ops_a = []
    ops_b = []
    ops_c = []

    # one-point terms x+y+z for A and B sublattices
    ops_Ax = []
    ops_Ay = []
    ops_Az = []
    ops_Bx = []
    ops_By = []
    ops_Bz = []

    exp_values = []

    # energy xx + yy + zz
    # We creat a mask that incompasses the largest support \Union(op_a, op_b)
    # such that the list of op_a matches that of op_b
    op_mask = [('Id', [0, 0], 1), ('Id', [0, 1], 0), ("Id", [1, 0], 0), ("Id", [1, 0], 1)]
    mps_inds = lattice.possible_multi_couplings(op_mask)[0]

    op_found = False
    for inds in mps_inds:
        if inds[0] in site:
            # use inds[0] as the site of reference for an operator
            op_found = True

            # xx+yy+zz
            op_a = [('Sigmaz', inds[0]), ('Sigmaz', inds[1])]  # i_A + z
            op_b = [('Sigmay', inds[0]), ('Sigmay', inds[2])]  # i_A + y
            op_c = [('Sigmax', inds[2]), ('Sigmax', inds[3])]  # i_A + x
            ops_a.append(op_a)
            ops_b.append(op_b)
            ops_c.append(op_c)

            # x+y+z
            op_Ax = [('Sigmax', inds[0])]
            op_Ay = [('Sigmay', inds[0])]
            op_Az = [('Sigmaz', inds[0])]

            op_Bx = [('Sigmax', inds[2])]
            op_By = [('Sigmay', inds[2])]
            op_Bz = [('Sigmaz', inds[2])]

            ops_Ax.append(op_Ax)
            ops_Ay.append(op_Ay)
            ops_Az.append(op_Az)
            ops_Bx.append(op_Bx)
            ops_By.append(op_By)
            ops_Bz.append(op_Bz)

    if not op_found:
        raise ValueError(site, "No available energy operator found according to the list of sites!")
    
    hx = model_params['Fx']
    hy = model_params['Fy']
    hz = model_params['Fz']
    print("-----------\n","Measuring the following energy density operators:")
    for op_a, op_b, op_c, op_Ax, op_Ay, op_Az, op_Bx, op_By, op_Bz in zip(ops_a, ops_b, ops_c, ops_Ax, ops_Ay, ops_Az, ops_Bx, ops_By, ops_Bz):
        print(op_a, " + ", op_b, "+", op_c, \
            f"+{hx:.2f}", op_Ax, f"+{hy:.2f}", op_Ay, f"+{hz:.2f}", op_Az, \
            f"+{hx:.2f}", op_Bx, f"+{hy:.2f}", op_By, f"+{hz:.2f}", op_Bz)
        expvalue = psi.expectation_value_term(op_a) + psi.expectation_value_term(op_b) + psi.expectation_value_term(op_c) \
                 + hx * psi.expectation_value_term(op_Ax) + hy * psi.expectation_value_term(op_Ay) + hz * psi.expectation_value_term(op_Az) \
                 + hx * psi.expectation_value_term(op_Bx) + hy * psi.expectation_value_term(op_By) + hz * psi.expectation_value_term(op_Bz)
        exp_values.append(expvalue) 
    

    return exp_values



def measure(measurements, env, M, model_params, site):
    """ function to measure several observables
    Parameters:
    measurements: thing to be measured
    evn: class MPSEnvironment(gs, psi)
    """
    # Attribute MPSEnvironment.ket() is inherited from BaseEnvironment; expectation_value() from class BaseMPSExpectationValue; Default is for the entire MPS
    # see https://github.com/tenpy/tenpy/blob/main/tenpy/networks/mps.py#L5389


    measurements['energy'] = measure_energy_densities(env.ket, M.lat, model_params, site)


    env.clear()


def run_statics(**kwargs):
    """ run time evolution and measure hole correlation functions
    kwargs:
        gs_file: the name of the file that contains the ground state and model parameters
        op_type: operator type, e.g. Sz, Sx, Sy
        j_unit_cell: location j of the operator, offset from the center
        dt: time step
        chi_max: bond dimension
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

    M = Kitaev_Extended(model_params)
    env = MPSEnvironment(gs, psi)
    chi_max = kwargs['chi_max']

    measurements = {}
    site = range(1, M.lat.N_sites, 2 * model_params['Ly'])
    measure(measurements, env, M, model_params, site)

    
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
        raise("missing arguments! 2 cmdargs gs_file and op_type!")
    cmdargs = sys.argv

    gs_file = cmdargs[1]
    chi_max = 900

    run_statics(gs_file=gs_file, chi_max=chi_max)