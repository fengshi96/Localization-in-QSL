"""
ground state simulations
"""
import numpy as np
from model_ladder import Kitaev_Ladder
import os
import h5py
from tenpy.tools import hdf5_io
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPOGraph, MPOEnvironment
from tenpy.networks.terms import TermList
from tenpy.algorithms import dmrg
from tenpy.algorithms.mps_common import SubspaceExpansion, DensityMatrixMixer
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ground_state(**kwargs):
    """
    full ground state simulation, where we fix plaquette terms
    """
    chi_max = kwargs['chi_max']
    Lx = kwargs['Lx']
    order = kwargs.get('order', 'default')
    J_K = kwargs['J_K']
    Fx = kwargs['Fx']
    Fy = kwargs['Fy']
    Fz = kwargs['Fz']
    bc = kwargs['bc']
    dmrg_restart = kwargs.get('dmrg_restart', False)
    gs_file_previous_run = kwargs.get('gs_file_previous_run', None)
    model_params = dict(Lx=Lx, order=order,
                        J_K=J_K, Fx=Fx, Fy=Fy, Fz=Fz, bc=bc, dmrg_restart=dmrg_restart)

    name = f'GS_L{Lx}{order}chi{chi_max}_K{J_K:.2f}Fx{Fx:.2f}Fy{Fy:.2f}Fz{Fz:.2f}'
    logger.info("name: "+name)

    # run DMRG
    M = Kitaev_Ladder(model_params)

    dmrg_params = {'mixer': DensityMatrixMixer,
                   'mixer_params': {
                    'amplitude': 1.e-6,
                    'decay': 1.,
                    'disable_after': 5,
                    },
                   'N_sweeps_check': 1,
                   'trunc_params': {
                    'chi_max': chi_max,
                    'svd_min': 1.e-12,
                    'max_trunc_err': 1.e-4
                    },
                    'max_E_err': 1.e-8,
                    'min_sweeps': 20,
                    'max_sweeps': 30,
                    'max_hours': 80,
                    'combine': True}

    product_state = ['up','down'] * Lx

    if dmrg_restart:
        print("dmrg_restart: I'm restarting dmrg from previous run!")
        with h5py.File(gs_file_previous_run, 'r') as f:
            state_previous = hdf5_io.load_from_hdf5(f)
        psi = state_previous['gs']
    else:
        psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS,  dtype='complex')

    if psi.finite:
        logger.info("quantum numbers are: {}".format(psi.get_total_charge(True)))
    info = dmrg.run(psi, M, dmrg_params)

    E = info["E"]
    logger.info(f"chi is {psi.chi}")
    logger.info(f"energy with fields: {E}")
    mps_indx2pos = M.positions()
    print(mps_indx2pos)

    state = {"gs": psi, "model_params": model_params, "dmrg_params": dmrg_params,
             "info": info, "pos": mps_indx2pos}

    # save file
    if not os.path.exists("./ground_states/"):
        os.makedirs("./ground_states/")
    # import h5py
    # from tenpy.tools import hdf5_io
    with h5py.File('./ground_states/'+name+'.h5', 'w') as f:
        hdf5_io.save_to_hdf5(f, state)

    corr = psi.correlation_function('Sigmaz', 'Sigmaz')
    corr = np.round(corr, 2)
    print(corr)




# ----------------------------------------------------------
# ----------------- Run run_ground_state() ----------------
# ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.argv ## get the input argument
    total = len(sys.argv)
    if total !=4:
        raise("missing arguments! 3 cmdargs Fx Fy Fz!")
    cmdargs = sys.argv

    chi_max = 80
    Lx = 42
    J_K = -1.0
    Fx = -float(cmdargs[1])
    Fy = -float(cmdargs[2])
    Fz = -float(cmdargs[3])
    order = 'folded' 
    bc = 'periodic'

    # uncomment if restart dmrg
    dmrg_restart = False
    gs_file_previous_run = "./ground_states/" + f'GS_L{Lx}{order}bc{bc}chi{chi_max}_K{J_K:.2f}Fx{Fx:.2f}Fy{Fy:.2f}Fz{Fz:.2f}' + '.h5'


    run_ground_state(chi_max=chi_max, Lx=Lx, order=order, bc = bc,
                        J_K=J_K, Fx=Fx, Fy=Fy, Fz=Fz, dmrg_restart=dmrg_restart, gs_file_previous_run=gs_file_previous_run)
