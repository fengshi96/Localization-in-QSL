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
from mpo_fluxes import S1Operators, S2Operators, fluxes_op
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def perturb_state_S1(psi, model_params, site=[0], chi_max=200):
    """ Perturb the state psi by applying a local energy density operator at site
    """
    op_params = model_params
    # del op_params['dmrg_restart']
    op_params['siteRef'] = site
    E_MPO = S1Operators(op_params).H_MPO 

    option = {
            'compression_method': 'zip_up', 'm_temp': 2, 'trunc_weight': 0.5,
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10, 'trunc_cut': 1e-10 }
            }
    E_MPO.apply(psi, option)  # this alters psi in place
    print("[perturb_state_S1] finished applying the flux operator S1 at site: ", site)



def perturb_state_S2(psi, model_params, site=[0], chi_max=200):
    """ Perturb the state psi by applying a local energy density operator at site
    """
    op_params = model_params
    # del op_params['dmrg_restart']
    op_params['siteRef'] = site
    E_MPO = S2Operators(op_params).H_MPO 

    option = {
            'compression_method': 'zip_up', 'm_temp': 2, 'trunc_weight': 0.5,
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10, 'trunc_cut': 1e-10 }
            }
    
    E_MPO.apply(psi, option)  # this alters psi in place
    print("[perturb_state_S2] finished applying the flux operator S2 at site: ", site)





def measure_fluxes(psi, lattice, site='all'):
    """
    Measure flux operators for a given psi.
    For a ladder, there are three types of fluxes: S1 (square faces a), S2 (square faces b) and Wp (hexagons).
    
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

    ops_s1, ops_s2, ops_wp = fluxes_op(lattice, site=site)
    exp_values_s1 = []
    exp_values_s2 = []
    exp_values_wp = []
    
    for op_s1 in ops_s1:
        # print(op_s1)
        expvalue = psi.expectation_value_term(op_s1) + 0.
        exp_values_s1.append(expvalue) 

    for op_s2 in ops_s2:
        # print(op_s2)
        expvalue = psi.expectation_value_term(op_s2) + 0.
        exp_values_s2.append(expvalue) 

    for op_wp in ops_wp:
        # print(op_wp)
        expvalue = psi.expectation_value_term(op_wp) + 0.
        exp_values_wp.append(expvalue) 
    
    return exp_values_s1, exp_values_s2, exp_values_wp



def measure_flux_correlators(psi, lattice, model_params, sites, siteRef):
    """
    Measure flux correlators for a given psi.
    For a ladder, we measure two types of fluxes: S1 (square faces a), S2 (square faces b).
    <0|X U'(t) S1_sites S1_siteRef  U(t) X|0>
    <0|X U'(t) S2_sites S2_siteRef  U(t) X|0>
    psi = U(t) X|0> is the state we want to measure the flux correlators
    """
    print("[measure_flux_correlators] measuring flux correlators")
    if sites == 'all':
        sites = range(lattice.N_sites)
    elif type(sites) == int:
        # if a single site as int, turn it into a list e.g. 1 -> [1]
        s_list = [sites, ]
        sites = s_list

    corr_values_s1 = []
    corr_values_s2 = []

    ops_s1, ops_s2, _ = fluxes_op(lattice, site=sites)


    ket = psi.copy()  # ket = U(t) X|0>
    env_temp = MPSEnvironment(psi, ket)
    perturb_state_S1(ket, model_params, site=siteRef) # S1_siteRef  U(t) X|0>
    for op_s1 in ops_s1:
        print(op_s1) 
        expvalue = env_temp.expectation_value_term(op_s1) + 0.  #  <0| X U'(t) S1_s S1_siteRef  U(t) X|0> for all s in sites
        corr_values_s1.append(expvalue) 


    ket = psi.copy()  # ket = U(t) X|0>
    env_temp = MPSEnvironment(psi, ket)
    perturb_state_S2(ket, model_params, site=siteRef) # S2_siteRef  U(t) X|0>
    for op_s2 in ops_s2:
        print(op_s2)
        expvalue = env_temp.expectation_value_term(op_s2) + 0.  #  <0| X U'(t) S2_s S2_siteRef  U(t) X|0> for all s in sites
        corr_values_s2.append(expvalue)

    env_temp.clear()
    return corr_values_s1, corr_values_s2


def measure_evolved_fluxes(measurements, evolved_time, env, M, model_params, siteRef, site='all'):
    """ function to measure flux operators during time-evolution
    Parameters:
    measurements: thing to be measured
    evolved_time: evolved time
    evn: class MPSEnvironment(gs, psi)
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
    if t_key == -1:  # save gs expval to t = -1
         measurements[t_key]['S1_gs'], measurements[t_key]['S2_gs'], measurements[t_key]['Wp_gs'] = measure_fluxes(env.ket, M.lat, site)
         measurements[t_key]['S1_correlation_gs'], measurements[t_key]['S2_correlation_gs'] = measure_flux_correlators(env.ket,  M.lat, model_params, site, siteRef)

    measurements[t_key]['entropy'] = env.ket.entanglement_entropy()
    measurements[t_key]['chi'] = env.ket.chi
    measurements[t_key]['S1_gs'], measurements[t_key]['S2_gs'], measurements[t_key]['Wp_gs'] = measurements[-1]['S1_gs'], measurements[-1]['S2_gs'], measurements[-1]['Wp_gs']
    measurements[t_key]['S1'], measurements[t_key]['S2'], measurements[t_key]['Wp'] = measure_fluxes(env.ket, M.lat, site)

    measurements[t_key]['delta_S1'] = [s1 - gs for s1, gs in zip(measurements[t_key]['S1'], measurements[t_key]['S1_gs'])]
    measurements[t_key]['delta_S2'] = [s2 - gs for s2, gs in zip(measurements[t_key]['S2'], measurements[t_key]['S2_gs'])]
    measurements[t_key]['delta_Wp'] = [wp - wg for wp, wg in zip(measurements[t_key]['Wp'], measurements[t_key]['Wp_gs'])]
    measurements[t_key]['S1_correlation_gs'], measurements[t_key]['S2_correlation_gs'] = measurements[-1]['S1_correlation_gs'], measurements[-1]['S2_correlation_gs']
    measurements[t_key]['S1_correlation'], measurements[t_key]['S2_correlation'] = measure_flux_correlators(env.ket,  M.lat, model_params, site, siteRef)
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
    time_state_name = kwargs.get('time_state_name', None)
    op_type = kwargs['op_type']
    model_params = state['model_params']
    hx = model_params['Fx']
    hy = model_params['Fy']
    hz = model_params['Fz']
    order = model_params.get('order', "default")
    print("order: ", order)

    if run_time_restart:
        time_state_name = kwargs['time_state_name']
        tsteps_init = 0
        tsteps_cont = kwargs['tsteps_cont'] # var for ExpMPO, again 2site for TDVP
        with h5py.File(time_state_name, 'r') as f:
            psi = hdf5_io.load_from_hdf5(f['last MPS'])
            gs = state['gs']
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
    M = Kitaev_Ladder(model_params)

    chi_max = kwargs['chi_max']
    save_state = kwargs.get('save_state', False)  # whether we want to save the full states at some time steps
    evolve_gs = kwargs.get('evolve_gs', False)  # whether we want to also time evolve the ground state
    dt = kwargs['dt']
    t_method = kwargs['t_method']
    assert t_method == 'TDVP' or t_method == 'ExpMPO'
    
    t_name = f"_Flux_chi{chi_max}dt{dt:.3f}{t_method}_"+name
    logger.info("name: " + t_name)

    # Parameters of ExpMPOEvolution(psi, model, t_params)
    # We set N_steps = 1, such that ExpMPOEvolution.run() evolves N_steps * dt = dt in time
    t_params = {'run_time_restart': run_time_restart, 'dt': dt, 'N_steps': 1, 'order': 2, 'approximation': 'II',
            'compression_method': 'zip_up', 'm_temp': 2, 'trunc_weight': 0.5,
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

        # define sites to measure energy density operators
        if model_params['bc'] == 'open':
            site = range(0, M.lat.N_sites, 4)
        else:
            site = [0, 8, 16, 14, 6]  # FIX ME Later!!!


        siteRef_measurement = model_params['Lx'] - 3
        print("siteRef: ", siteRef_measurement)
        measure_evolved_fluxes(measurements, -1.00, env, M, model_params, siteRef_measurement, site=site) # label gs as evolved_time = -1 to avoid confusion with evolved states

        # if it is a brand new time-evolution from t=0
        if not run_time_restart:
            # then perturb by Ops at the site before time evolution by psi = \prod_j op_j |gs>
            print("Achtung! I'm directly perturbing on the sites!")
            t_name = "_Pert_" + op_type + t_name
            j = model_params['Lx'] - 1
            print("Perturbing site at: ", j)
            psi.apply_local_op(j, op_type, unitary=False, renormalize=True)

        else:
            print("Restart time evolution!")
            t_name = "_Restart_Pert_" + op_type + t_name


        # measure the ground state expectation of energy
        if not run_time_restart:
            # measurements['gs_energy'] = measure_energy_densities(env.ket, M.lat, model_params, site=range(1, M.lat.N_sites, 2 * model_params['Ly']))
            measure_evolved_fluxes(measurements, eng.evolved_time, env, M, model_params, siteRef_measurement, site=site)

        # exp(-iHt) |psi> = exp(-iHt) Op_j0 |gs>
        for i in range(tsteps_init):
            t0 = time.time()
            eng.run()
            if evolve_gs:
                eng_gs.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved_fluxes(measurements, eng.evolved_time, env, M, model_params, siteRef_measurement, site=site)
            logger.info(f"measurement took {(time.time()-t2):.3f}s")
            logger.info(f"t = {eng.evolved_time}")
            logger.info("---------------------------------")
        t_params['start_time'] = eng.evolved_time  # update clock for the following evolution

        # switch to other engine
        t_params['compression_method'] = 'variational'
        eng = ExpMPOEvolution(psi, M, t_params)
        if evolve_gs:
            eng_gs = ExpMPOEvolution(gs, M, t_params)


        if not os.path.exists("./flux_time_states/"):
            os.makedirs("./flux_time_states/")

        for i in range(tsteps_cont):
            t0 = time.time()
            eng.run()
            if evolve_gs:
                eng_gs.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved_fluxes(measurements, eng.evolved_time, env, M, model_params, siteRef_measurement, site=site)
            logger.info(f"measurement took {(time.time()-t2):.3f}s")
            logger.info(f"t = {eng.evolved_time}")
            logger.info("---------------------------------")
            print(measurements)

            # save state at the last step
            if i%100 == 0 or i == tsteps_cont - 1:  # i == tsteps_cont - 1: #i%2 == 0:
                state_t = {"gs_file": gs_file, "t_params": t_params, "fields": str(hx)+str(hy)+str(hz), "op_type": op_type, "save_state": save_state, "order": order,
                           "chi_max": chi_max, "dt": dt, "t_method": t_method, "evolved_time": eng.evolved_time, "last MPS": psi.copy(), "model_params": model_params, "pos": M.positions(),
                           "measurements": measurements}
                if save_state:
                    with h5py.File(f'./flux_time_states/{tsteps_init+previous_evolved_steps+i+1}'+f'time{round(eng.evolved_time, 2)}'+t_name, 'w') as f:
                        hdf5_io.save_to_hdf5(f, state_t)


    else:
        raise ValueError('So far only ExpMPO method is implemented')
    



# ----------------------------------------------------------
# ----------------- Debug run_time() ----------------
# ----------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.argv ## get the input argument

    run_time_restart = False
    
    total = len(sys.argv)

    if total < 2:
        raise("missing arguments! at least 1 cmdargs gs_file!")
    cmdargs = sys.argv

    gs_file = "./ground_states/GS_L11defaultchi90_K-1.00Fx-0.00Fy-0.00Fz-0.00.h5" 
    # gs_file = cmdargs[1]

    previous_run_file = None
    if run_time_restart:
        previous_run_file = './energy_time_states/3time0.03_Pert_Sm_Energy_chi350dt0.010ExpMPO_GS_L33Cstylechi350_K-1.00Fx0.00Fy0.00Fz0.00W0.00EpFalse.h5'

    op_type = "Sigmaz"
    dt = 0.01
    t_method = "ExpMPO"
    tsteps_init = 1
    tsteps_cont = 1
    save_state = True
    evolve_gs = False
    chi_max = 100

    

    run_time(gs_file=gs_file, chi_max=chi_max, op_type=op_type, dt=dt, t_method=t_method, tsteps_cont=tsteps_cont, tsteps_init=tsteps_init,
             save_state=save_state, evolve_gs=evolve_gs, run_time_restart=run_time_restart, time_state_name=previous_run_file)
