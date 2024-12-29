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


def measure_Wp(psi, lattice, site='all'):
    """
    Measure Wp plaquette operators.
    """
    if site == 'all':
        site = range(psi.L)
    elif type(site) == int:
        s_list = [site, ]
        site = s_list

    exp_values = []

    ops = [('Sz', [0, 0], 0), ('Sx', [0, 0], 1), ('Sy', [1, 0], 0), ('Sz', [1, -1], 1), ('Sx', [1, -1], 0), ('Sy', [0, -1], 1)]
    mps_inds = lattice.possible_multi_couplings(ops)[0]
    for inds in mps_inds:
        if inds[0] in site:
            ops = [('Sz', inds[0]), ('Sx', inds[1]), ('Sy', inds[2]), ('Sz', inds[3]), ('Sx', inds[4]), ('Sy', inds[5])]
            exp_values.append(psi.expectation_value_term(ops)*2**6)

    return np.array(exp_values)


def measure_current_densities(cur_op, psi, lattice, site='all'):
    """
    Measure current density operators.
    
    Inputs:
    cur_op: current operator names, Je for energy current or Js for spin current
    psi: time-dependent state exp(-iHt)|perturbed> modified in-place in mps environment
    lattice: Attribute lattice of the model, e.g. Honeycomb.lat
    site: list of sites in which available current density operators are matched in terms of operators' site of reference

    Output:
    expectation value <psi| cur_op |psi>
    """
    if site == 'all':
        site = range(lattice.N_sites)
    elif type(site) == int:
        # if a single site as int, turn it into a list e.g. 1 -> [1]
        s_list = [site, ]
        site = s_list

    ops_a = []
    ops_b = []
    exp_values = []

    if cur_op == 'Je':
        # Je = op_a - op_b
        # We creat a mask that incompasses the largest support \Union(op_a, op_b)
        # such that the list of op_a matches that of op_b
        op_mask = [('Id', [0, 0], 0), ('Id', [0, 0], 1), ("Id", [0, 1], 0), ("Id", [1, 0], 0)]
        mps_inds = lattice.possible_multi_couplings(op_mask)[0]

        op_found = False
        for inds in mps_inds:
            if inds[0] in site:
                # use inds[0] as the site of reference for an operator
                op_found = True
                op_a = [('Sigmay', inds[0]), ('Sigmaz', inds[1]), ('Sigmax', inds[2])]
                op_b = [('Sigmay', inds[0]), ('Sigmax', inds[1]), ('Sigmaz', inds[3])]
                ops_a.append(op_a)
                ops_b.append(op_b)
            
        if not op_found:
            raise ValueError(site, "No available Je operator found according to the list of sites!")
       
        print("-----------\n","Measuring the following energy current density operators:")
        for op_a, op_b in zip(ops_a, ops_b):
            print(op_a, " - ", op_b)
            expvalue = psi.expectation_value_term(op_a) - psi.expectation_value_term(op_b)
            exp_values.append(expvalue) 
    
    elif cur_op == 'Js':
        # Js = op_a - op_b
        op_mask = [('Id', [0, 0], 0), ('Id', [0, 0], 1)]
        mps_inds = lattice.possible_multi_couplings(op_mask)[0]

        op_found = False
        for inds in mps_inds:
            if inds[0] in site:
                # use inds[0] as the site of reference for an operator
                op_found = True
                op_a = [('Sigmax', inds[0]), ('Sigmay', inds[1])]
                op_b = [('Sigmay', inds[0]), ('Sigmax', inds[1])]
                ops_a.append(op_a)
                ops_b.append(op_b)
         
        if not op_found:
            raise ValueError(site, "No available Js operator found according to the list of sites!")
       
        print("-----------\n","Measuring the following spin current density operators:")
        for op_a, op_b in zip(ops_a, ops_b):
            print(op_a, " - ", op_b)
            expvalue = psi.expectation_value_term(op_a) - psi.expectation_value_term(op_b)
            exp_values.append(expvalue)

    else:
        raise("current operator not recognized")

    return exp_values


if __name__ == "__main__":
    Lx = 4; Ly = 3
    order = 'Cstyle'
    J_K = 1; J_H = 0
    Jnnn = 0; G = 0
    G_P = 0; W = 0; bc = ['open', 'periodic']
    model_params = dict(Lx=Lx, Ly=Ly, order=order,
                        J_K=J_K, J_H=J_H, Jnnn=Jnnn, G=G, G_P=G_P, W=W)
    M = Kitaev_Extended(model_params)

    gs_file = 'test/GS_L43Cstylechi350_Jx1.00Jy1.00Jz1.00JH0.37Jnnn-0.30G-0.25GP-0.27Fx0.00Fy0.00Fz0.00Wp0.00.h5'
#     gs_file = 'GS_L43Cstylechi350_Jx-1.00Jy-1.00Jz-1.00JH0.00Jnnn0.00G0.00GP0.00Fx0.00Fy0.00Fz0.00Wp0.00.h5'
    with h5py.File('./' + gs_file, 'r') as f:
        state = hdf5_io.load_from_hdf5(f)

    psi = state['gs']
#     for i in range(6):
#         psi.apply_local_op(i, 'Sx', unitary=False, renormalize=True)

    exp_values = measure_current_densities('Js', psi=psi, lattice=M.lat, site=range(0, M.lat.N_sites, 2 * model_params['Ly']))
    print(exp_values)

    for j in range(0, 2 * model_params['Ly']):
        print(j)














