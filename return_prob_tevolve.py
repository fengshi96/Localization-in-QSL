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


def measure_evolved_correlation(measurements, evolved_time, env, E0, M, op_type):
    """ function to measure correlations during time-evolution
    Parameters:
    measurements: thing to be measured
    evolved_time: evolved time
    evn: class MPSEnvironment(gs, psi)
    E0: ground state energy
    M: class Model
    
    Output Example:
    measurements = {
        t_key = 0.0: {
            'entropy': [...],
            'chi': [...],
            'Wp': value,
            'Cij_Sz': value,
        },
        t_key = 0.1: {
            'entropy': [...],
            'chi': [...],
            'Wp': value,
            'Cij_Sz': value,
        },
        ...
    }
    """
    # the time label at t = evolved_time
    t_key = np.round(evolved_time, 3)
    # creates a new dictionary for the current time t_key to store measurements. measurements[t_key] is a dictionary of the dicts
    print("-------------------- t_key or evolved time == ", t_key)
    measurements[t_key] = {}

    # Attribute MPSEnvironment.ket() is inherited from BaseEnvironment; expectation_value() from class BaseMPSExpectationValue; Default is for the entire MPS
    # see https://github.com/tenpy/tenpy/blob/main/tenpy/networks/mps.py#L5389
    measurements[t_key]['entropy'] = env.ket.entanglement_entropy()
    measurements[t_key]['chi'] = env.ket.chi
    measurements[t_key][op_type] = env.ket.expectation_value(op_type)
    measurements[t_key][f'Cij_{op_type}'] = np.exp(1j*evolved_time*E0) * env.expectation_value(f'{op_type}')
    
    env.clear()


def run_time(**kwargs):
    """ run time evolution and measure hole correlation functions
    kwargs:
        gs_file: the name of the file that contains the ground state and model parameters
        op_type: operator type, e.g. Sigmaz, Sigmax, Sigmay
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
    name = os.path.basename(gs_file)
    with h5py.File(gs_file, 'r') as f:
        state = hdf5_io.load_from_hdf5(f)

    j_unit_cell = kwargs['j_unit_cell']
    op_type = kwargs['op_type']
    model_params = state['model_params']
    hx = model_params['Fx']
    hy = model_params['Fy']
    hz = model_params['Fz']
    gs = state['gs']
    psi = gs.copy()
    Lx = model_params['Lx']
    Ly = model_params['Ly']
    order = model_params.get('order', "default")
    model_params['W'] = 0

    M = Kitaev_Ladder(model_params)

    E0 = MPOEnvironment(gs, M.H_MPO, gs).full_contraction(0)

    chi_max = kwargs['chi_max']
    save_state = kwargs.get('save_state', False)  # whether we want to save the full states at some time steps
    evolve_gs = kwargs.get('evolve_gs', False)  # whether we want to also time evolve the ground state
    dt = kwargs['dt']
    t_method = kwargs['t_method']
    assert t_method == 'TDVP' or t_method == 'ExpMPO'
    
    if evolve_gs:
        t_name = f"_Corr_evoGS_j{j_unit_cell}op{op_type}chi{chi_max}dt{dt:.3f}{t_method}_"+name
        E0 = 0  # don't need phase in correlations
    else:
        t_name = f"_Corr_j{j_unit_cell}op{op_type}chi{chi_max}dt{dt:.3f}{t_method}_"+name
    logger.info("name: " + t_name)

    tsteps_init = kwargs['tsteps_init'] # SVDs for ExpMPO, 2site for TDVP
    tsteps_cont = kwargs['tsteps_cont'] # var for ExpMPO, again 2site for TDVP

    # Parameters of ExpMPOEvolution(psi, model, t_params)
    # We set N_steps = 1, such that ExpMPOEvolution.run() evolves N_steps * dt = dt in time
    t_params = {'dt': dt, 'N_steps': 1, 'order': 2, 'approximation': 'II',
            'compression_method': 'zip_up', 'm_temp': 5, 'trunc_weight': 0.1,
            'tsteps_init': tsteps_init, 'tsteps_cont': tsteps_cont,
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10, 'trunc_cut': None, }
            }

    measurements = {}

    if t_method == 'ExpMPO':
        # now psi = gs.copy()
        eng = ExpMPOEvolution(psi, M, t_params)
        if evolve_gs:
            eng_gs = ExpMPOEvolution(gs, M, t_params)
        # stores the partial contractions between gs(bra) and psi(ket) up to each bond.
        # note that psi will be modified in-place
        env = MPSEnvironment(gs, psi)
        
        if order == "default":
            j_0 = int(2*Lx*Ly/2 + j_unit_cell*Ly)  # because of 2-site unit cell
        elif order == "Cstyle":
            j_0 = int(2*Lx*Ly/2 + j_unit_cell)
            # e.g. if j_unit_cell = 0, the site of reference would be at (x/2, 0) in Bravais lattice

        # Apply Op at j0 onto the gs before time evolution
        # psi = Op_j0 |gs>
        psi.apply_local_op(j_0, op_type, unitary=False, renormalize=True)

        # measure the ground state expectation of Op_j0
        measure_evolved_correlation(measurements, eng.evolved_time, env, E0, M, op_type)

        # exp(-iHt) |psi> = exp(-iHt) Op_j0 |gs>
        for i in range(tsteps_init):
            t0 = time.time()
            eng.run()
            if evolve_gs:
                eng_gs.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved_correlation(measurements, eng.evolved_time, env, E0, M, op_type)
            logger.info(f"measurement took {(time.time()-t2):.3f}s")
            logger.info(f"t = {eng.evolved_time}")
            logger.info("---------------------------------")
        t_params['start_time'] = eng.evolved_time  # update clock for the following evolution

        # switch to other engine
        t_params['compression_method'] = 'variational'
        eng = ExpMPOEvolution(psi, M, t_params)
        if evolve_gs:
            eng_gs = ExpMPOEvolution(gs, M, t_params)

        
        if not os.path.exists("./time_states/"):
            os.makedirs("./time_states/")

        for i in range(tsteps_cont):
            t0 = time.time()
            eng.run()
            if evolve_gs:
                eng_gs.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved_correlation(measurements, eng.evolved_time, env, E0, M, op_type)
            logger.info(f"measurement took {(time.time()-t2):.3f}s")
            logger.info(f"t = {eng.evolved_time}")
            logger.info("---------------------------------")
            print(measurements)

            # save state at the last step
            t_name = "_" + op_type + t_name
            if i%10 == 0: #i%2 == 0: i == tsteps_cont - 1:
                state_t = {"gs_file": gs_file, "t_params": t_params, "fields": str(hx)+str(hy)+str(hz), "j_unit_cell": j_unit_cell, "op_type": op_type, "save_state": save_state, "evolve_gs": evolve_gs,
                           "chi_max": chi_max, "dt": dt, "t_method": t_method, "evolved_time": eng.evolved_time, "last MPS": psi.copy(),
                           "last GS": gs.copy(), "model_params": model_params, "pos": M.positions(), 
                           "measurements": measurements}
                if save_state:
                    with h5py.File(f'./time_states/{tsteps_init+i+1}'+f'time{eng.evolved_time}'+t_name, 'w') as f:
                        hdf5_io.save_to_hdf5(f, state_t)


    else:
        raise ValueError('So far only ExpMPO method is implemented')
    



# ----------------------------------------------------------
# ----------------- Debug run_time() ----------------
# ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.argv ## get the input argument
    total = len(sys.argv)
    if total !=3:
        raise("missing arguments! 2 cmdargs gs_file and op_type!")
    cmdargs = sys.argv

    gs_file = cmdargs[1]
    op_type = cmdargs[2]
    j_unit_cell = 2  # A sublat at Ly // 2
    dt = 0.01
    t_method = "ExpMPO"
    tsteps_init = 20
    tsteps_cont = 180
    save_state = True
    evolve_gs = False
    chi_max = 800

    run_time(gs_file=gs_file, chi_max=chi_max, op_type=op_type, j_unit_cell=j_unit_cell, dt=dt, t_method=t_method, tsteps_cont=tsteps_cont, tsteps_init=tsteps_init,save_state=save_state, evolve_gs=evolve_gs)