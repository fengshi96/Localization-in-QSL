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

def measure_current_densities(cur_op, psi, lattice, site='all'):
    """
    Measure current density operators for a given psi.
    
    Inputs:
    cur_op: current operator names, Je for energy current or Js for spin current
    psi: any state, e.g. a time-dependent state exp(-iHt)|perturbed> modified in-place in mps environment
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
                op_a = [('Sigmax', inds[0]), ('Sigmay', inds[1]), ('Sigmaz', inds[2])]
                op_b = [('Sigmax', inds[0]), ('Sigmaz', inds[1]), ('Sigmay', inds[3])]
                # op_a = [('Sigmay', inds[0]), ('Sigmaz', inds[1]), ('Sigmax', inds[2])]
                # op_b = [('Sigmay', inds[0]), ('Sigmax', inds[1]), ('Sigmaz', inds[3])]
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



def measure_evolved_current(measurements, evolved_time, env, M, site='all'):
    """ function to measure current density operators during time-evolution
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

    measurements[t_key]['entropy'] = env.ket.entanglement_entropy()
    measurements[t_key]['chi'] = env.ket.chi
    measurements[t_key]['Je'] = measure_current_densities('Je', env.ket, M.lat, site)
    measurements[t_key]['Js'] = measure_current_densities('Js', env.ket, M.lat, site)

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
    order = model_params.get('order', "Cstyle")
    model_params['W'] = 0

    if run_time_restart:
        time_state_name = kwargs['time_state_name']
        tsteps_init = 0
        tsteps_cont = kwargs['tsteps_cont'] # var for ExpMPO, again 2site for TDVP
        with h5py.File(time_state_name, 'r') as f:
            psi = hdf5_io.load_from_hdf5(f['last MPS'])
            gs = hdf5_io.load_from_hdf5(f['last GS'])
            measurements = hdf5_io.load_from_hdf5(f['measurements'])
            previous_evolved_time = hdf5_io.load_from_hdf5(f['evolved_time'])
            dt = hdf5_io.load_from_hdf5(f['dt'])
            previous_evolved_steps = int(previous_evolved_time/dt)
    else:
        tsteps_init = kwargs['tsteps_init'] # SVDs for ExpMPO, 2site for TDVP
        tsteps_cont = kwargs['tsteps_cont'] # var for ExpMPO, again 2site for TDVP
        previous_evolved_time = 0
        previous_evolved_steps = 0
        gs = state['gs']
        psi = gs.copy()
        measurements = {}
    
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
        t_name = f"_Current_chi{chi_max}dt{dt:.3f}{t_method}_"+name
        E0 = 0  # don't need phase in correlations
    else:
        t_name = f"_Current_chi{chi_max}dt{dt:.3f}{t_method}_"+name
    logger.info("name: " + t_name)

    # Parameters of ExpMPOEvolution(psi, model, t_params)
    # We set N_steps = 1, such that ExpMPOEvolution.run() evolves N_steps * dt = dt in time
    t_params = {'run_time_restart': run_time_restart, 'dt': dt, 'N_steps': 1, 'order': 2, 'approximation': 'II',
            'compression_method': 'zip_up', 'm_temp': 5, 'trunc_weight': 0.1,
            'tsteps_init': tsteps_init, 'tsteps_cont': tsteps_cont,
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10, 'trunc_cut': 1e-10 }
            }

    if t_method == 'ExpMPO':
        # now psi = gs.copy()
        eng = ExpMPOEvolution(psi, M, t_params)
        if run_time_restart:
            eng.evolved_time += previous_evolved_time

        if evolve_gs:
            eng_gs = ExpMPOEvolution(gs, M, t_params)
        # stores the partial contractions between gs(bra) and psi(ket) up to each bond.
        # note that psi will be modified in-place
        env = MPSEnvironment(gs, psi)

        # if it is a brand new time-evolution from t=0
        if not run_time_restart:
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
        else:
            print("Restart time evolution!")
            t_name = "_Restart_Pert_" + op_type + t_name

        # measure the ground state expectation of currents
        if not run_time_restart:
            measure_evolved_current(measurements, eng.evolved_time, env, M, site=range(0, M.lat.N_sites, 2 * model_params['Ly']))

        # exp(-iHt) |psi> = exp(-iHt) Op_j0 |gs>
        for i in range(tsteps_init):
            t0 = time.time()
            eng.run()
            if evolve_gs:
                eng_gs.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved_current(measurements, eng.evolved_time, env, M, site=range(0, M.lat.N_sites, 2 * model_params['Ly']))
            logger.info(f"measurement took {(time.time()-t2):.3f}s")
            logger.info(f"t = {eng.evolved_time}")
            logger.info("---------------------------------")
        t_params['start_time'] = eng.evolved_time  # update clock for the following evolution

        # switch to other engine
        t_params['compression_method'] = 'variational'
        eng = ExpMPOEvolution(psi, M, t_params)
        if evolve_gs:
            eng_gs = ExpMPOEvolution(gs, M, t_params)


        if not os.path.exists("./current_time_states/"):
            os.makedirs("./current_time_states/")

        for i in range(tsteps_cont):
            t0 = time.time()
            eng.run()
            if evolve_gs:
                eng_gs.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved_current(measurements, eng.evolved_time, env, M, site=range(0, M.lat.N_sites, 2 * model_params['Ly']))
            logger.info(f"measurement took {(time.time()-t2):.3f}s")
            logger.info(f"t = {eng.evolved_time}")
            logger.info("---------------------------------")
            print(measurements)

            # save state at the last step
            if i%10 == 0 or i == tsteps_cont - 1:  # i == tsteps_cont - 1: #i%2 == 0:
                state_t = {"gs_file": gs_file, "t_params": t_params, "fields": str(hx)+str(hy)+str(hz), "op_type": op_type, "save_state": save_state, "evolve_gs": evolve_gs,
                           "chi_max": chi_max, "dt": dt, "t_method": t_method, "evolved_time": eng.evolved_time, "last MPS": psi.copy(),
                           "last GS": gs.copy(), "model_params": model_params, "pos": M.positions(),
                           "measurements": measurements}
                if save_state:
                    with h5py.File(f'./current_time_states/{tsteps_init+previous_evolved_steps+i+1}'+f'time{round(eng.evolved_time, 2)}'+t_name, 'w') as f:
                        hdf5_io.save_to_hdf5(f, state_t)


    else:
        raise ValueError('So far only ExpMPO method is implemented')
    



# ----------------------------------------------------------
# ----------------- Debug run_time() ----------------
# ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.argv ## get the input argument

    run_time_restart = True
    
    total = len(sys.argv)

    if total < 2:
        raise("missing arguments! at least 1 cmdargs gs_file!")
    cmdargs = sys.argv

    # "./ground_states/GS_L33Cstylechi350_K-1.00Fx0.00Fy0.00Fz0.00W0.00EpFalse.h5" 
    gs_file = cmdargs[1]

    if run_time_restart:
        previous_run_file = './current_time_states/3time0.03_Pert_Sp_Current_chi350dt0.010ExpMPO_GS_L33Cstylechi350_K-1.00Fx0.00Fy0.00Fz0.00W0.00EpFalse.h5'

    op_type = "Sp"
    dt = 0.01
    t_method = "ExpMPO"
    tsteps_init = 1
    tsteps_cont = 2
    save_state = True
    evolve_gs = False
    chi_max = 350

    

    run_time(gs_file=gs_file, chi_max=chi_max, op_type=op_type, dt=dt, t_method=t_method, tsteps_cont=tsteps_cont, tsteps_init=tsteps_init,
             save_state=save_state, evolve_gs=evolve_gs, run_time_restart=run_time_restart, time_state_name=previous_run_file)
