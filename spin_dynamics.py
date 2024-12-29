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

    ops = [('Sigmaz', [0, 0], 0), ('Sigmax', [0, 0], 1), ('Sigmay', [1, 0], 0), ('Sigmaz', [1, -1], 1), ('Sigmax', [1, -1], 0), ('Sigmay', [0, -1], 1)]
    mps_inds = lattice.possible_multi_couplings(ops)[0]
    for inds in mps_inds:
        if inds[0] in site:
            ops = [('Sigmaz', inds[0]), ('Sigmax', inds[1]), ('Sigmay', inds[2]), ('Sigmaz', inds[3]), ('Sigmax', inds[4]), ('Sigmay', inds[5])]
            exp_values.append(psi.expectation_value_term(ops))
            print("Measured Wp: ", ops)

    return np.array(exp_values)


def measure_spin(op, psi, lattice, site='all'):
    if site == 'all':
        site = range(lattice.N_sites)
    elif type(site) == int:
        # if a single site as int, turn it into a list e.g. 1 -> [1]
        s_list = [site, ]
        site = s_list

    ops_a = []
    ops_b = []
    exp_values = []

    op_mask = [('Id', [0, 0], 0), ('Id', [0, 0], 1)]
    mps_inds = lattice.possible_multi_couplings(op_mask)[0]

    op_found = False
    for inds in mps_inds:
        if inds[0] in site:
            # use inds[0] as the site of reference for an operator
            op_found = True
            op_a = [('Sigmaz', inds[0])]
            op_b = [('Sigmaz', inds[1])]
            ops_a.append(op_a)
            ops_b.append(op_b)
        
    if not op_found:
        raise ValueError(site, "No available Je operator found according to the list of sites!")
    
    print("\n---------[measure_evolved] Meaure spin sites: ")
    for op_a, op_b in zip(ops_a, ops_b):
        print(op_a, " + ", op_b)
        expvalue = psi.expectation_value_term(op_a) + psi.expectation_value_term(op_b)
        exp_values.append(expvalue) 
    return exp_values


def measure_evolved(measurements, evolved_time, env, M, site='all'):
    """ function to measure spin operators during time-evolution
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
            'Js': value,
            'Je': value
        },
        t_key = 0.1: {
            'entropy': [...],
            'chi': [...],
            'Js': value,
            'Je': value
        },
        ...
    }
    """
    t_key = np.round(evolved_time, 3)
    print("-------------------- t_key or evolved time == ", t_key)
    measurements[t_key] = {}
    print("[measure_evolved] Meaure Wp sites: ")
    measurements[t_key]['Wp'] = measure_Wp(env.ket, M.lat, site)
    measurements[t_key]['Sigmaz'] = measure_spin('Sigmaz', env.ket, M.lat, site) # env.ket.expectation_value('Sigmaz', site)
    # [print(i) for i in site]
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

    run_time_restart = kwargs.get('run_time_restart', False)
    op_type = kwargs['op_type']
    model_params = state['model_params']
    hx = model_params['Fx']
    hy = model_params['Fy']
    hz = model_params['Fz']
    gs = state['gs']
    psi = gs.copy()
    order = model_params.get('order', "Cstyle")
    model_params['W'] = 0
    
    # In case the gs is prepared polarized on the left edge,
    # we then focus on the quench dynamics with the Hamiltonian without the edge polarizing field
    edge_polarization = False
    M = Kitaev_Extended(model_params)

    chi_max = kwargs['chi_max']
    save_state = kwargs.get('save_state', False)  # whether we want to save the full states at some time steps
    evolve_gs = kwargs.get('evolve_gs', False)  # whether we want to also time evolve the ground state
    dt = kwargs['dt']
    t_method = kwargs['t_method']
    assert t_method == 'TDVP' or t_method == 'ExpMPO'
    
    if evolve_gs:
        t_name = f"_Spin_chi{chi_max}dt{dt:.3f}{t_method}_"+name
        E0 = 0  # don't need phase in correlations
    else:
        t_name = f"_Spin_chi{chi_max}dt{dt:.3f}{t_method}_"+name
    logger.info("name: " + t_name)

    tsteps_init = kwargs['tsteps_init'] # SVDs for ExpMPO, 2site for TDVP
    tsteps_cont = kwargs['tsteps_cont'] # var for ExpMPO, again 2site for TDVP

    # Parameters of ExpMPOEvolution(psi, model, t_params)
    # We set N_steps = 1, such that ExpMPOEvolution.run() evolves N_steps * dt = dt in time
    t_params = {'dt': dt, 'N_steps': 1, 'order': 2, 'approximation': 'II',
            'compression_method': 'zip_up', 'm_temp': 5, 'trunc_weight': 0.1,
            'tsteps_init': tsteps_init, 'tsteps_cont': tsteps_cont,
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10, 'trunc_cut': 1e-10 }
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

        # get if gs is prepared with left edge polarized
        edge_polarization = model_params.get('edge_polarization')
        if not edge_polarization:
            # if the edge is not prepared polarized
            # then perturb by Ops at the left boundary before time evolution by psi = \prod_j op_j |gs>
            print("Achtung! I'm directly perturbing on the left edge!")
            t_name = "_Pert_" + op_type + t_name
            for j in range(0, 2 * model_params['Ly']):
                psi.apply_local_op(j, op_type, unitary=False, renormalize=True)
        else:
            print("Achtung! The left edge was prepared polarized! I'm now doing quench dynamics without the polarizing field!")

        # measure the ground state expectation of currents
        measure_evolved(measurements, eng.evolved_time, env, M, site=range(0, M.lat.N_sites, 2 * model_params['Ly']))

        # exp(-iHt) |psi> = exp(-iHt) Op_j0 |gs>
        for i in range(tsteps_init):
            t0 = time.time()
            eng.run()
            if evolve_gs:
                eng_gs.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved(measurements, eng.evolved_time, env, M, site=range(0, M.lat.N_sites, 2 * model_params['Ly']))
            logger.info(f"measurement took {(time.time()-t2):.3f}s")
            logger.info(f"t = {eng.evolved_time}")
            logger.info("---------------------------------")
        t_params['start_time'] = eng.evolved_time  # update clock for the following evolution

        # switch to other engine
        t_params['compression_method'] = 'variational'
        eng = ExpMPOEvolution(psi, M, t_params)
        if evolve_gs:
            eng_gs = ExpMPOEvolution(gs, M, t_params)


        if not os.path.exists("./spin_time_states/"):
            os.makedirs("./spin_time_states/")

        for i in range(tsteps_cont):
            t0 = time.time()
            eng.run()
            if evolve_gs:
                eng_gs.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved(measurements, eng.evolved_time, env, M, site=range(0, M.lat.N_sites, 2 * model_params['Ly']))
            logger.info(f"measurement took {(time.time()-t2):.3f}s")
            logger.info(f"t = {eng.evolved_time}")
            logger.info("---------------------------------")
            print(measurements)

            # save state at the last step
            if i%100== 0 or i == tsteps_cont - 1:  # i == tsteps_cont - 1: #i%2 == 0:
                state_t = {"gs_file": gs_file, "t_params": t_params, "fields": str(hx)+str(hy)+str(hz), "op_type": op_type, "save_state": save_state, "evolve_gs": evolve_gs,
                           "chi_max": chi_max, "dt": dt, "t_method": t_method, "evolved_time": eng.evolved_time, "last MPS": psi.copy(),
                           "last GS": gs.copy(), "model_params": model_params, "pos": M.positions(),
                           "measurements": measurements}
                if save_state:
                    with h5py.File(f'./spin_time_states/{tsteps_init+i+1}'+f'time{round(eng.evolved_time, 2)}'+t_name, 'w') as f:
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
    if total !=2:
        raise("missing arguments! 1 cmdargs gs_file!")
    cmdargs = sys.argv

    gs_file = cmdargs[1] # "./ground_states/GS_L43Cstylechi350_K-1.00Fx0.05Fy0.05Fz0.05W0.00.h5" 
    op_type = "Sp"
    dt = 0.01
    t_method = "ExpMPO"
    tsteps_init = 20
    tsteps_cont = 980
    save_state = True
    evolve_gs = False
    chi_max = 600

    run_time(gs_file=gs_file, chi_max=chi_max, op_type=op_type, dt=dt, t_method=t_method, tsteps_cont=tsteps_cont, tsteps_init=tsteps_init,save_state=save_state, evolve_gs=evolve_gs)