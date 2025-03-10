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



def measure_evolved_energy(measurements, evolved_time, env, M, model_params, site='all'):
    """ function to measure energy density operators during time-evolution
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
    measurements[t_key]['energy'] = measure_energy_densities(env.ket, M.lat, model_params, site)

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
    
    if evolve_gs:
        t_name = f"_Energy_chi{chi_max}dt{dt:.3f}{t_method}_"+name
        E0 = 0  # don't need phase in correlations
    else:
        t_name = f"_Energy_chi{chi_max}dt{dt:.3f}{t_method}_"+name
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

        measure_evolved_energy(measurements, -1.00, env, M, model_params, site=site) # label gs as evolved_time = -1 to avoid confusion with evolved states

        # if it is a brand new time-evolution from t=0
        if not run_time_restart:
            # then perturb by Ops at the site before time evolution by psi = \prod_j op_j |gs>
            print("Achtung! I'm directly perturbing on the sites!")
            t_name = "_Pert_" + op_type + t_name
            if model_params['bc'] == 'open':
                for j in np.array([model_params['Lx'] - 1, model_params['Lx']]):
                    print("Perturbing site ", j)
                    psi.apply_local_op(j, op_type, unitary=False, renormalize=True)
            else:
                for j in np.array([2 * model_params['Lx'] - 2, 2 * model_params['Lx'] - 1]):
                    print("Perturbing site ", j)
                    psi.apply_local_op(j, op_type, unitary=False, renormalize=True)

        else:
            print("Restart time evolution!")
            t_name = "_Restart_Pert_" + op_type + t_name


        # measure the ground state expectation of energy
        if not run_time_restart:
            # measurements['gs_energy'] = measure_energy_densities(env.ket, M.lat, model_params, site=range(1, M.lat.N_sites, 2 * model_params['Ly']))
            measure_evolved_energy(measurements, eng.evolved_time, env, M, model_params, site=site)

        # exp(-iHt) |psi> = exp(-iHt) Op_j0 |gs>
        for i in range(tsteps_init):
            t0 = time.time()
            eng.run()
            if evolve_gs:
                eng_gs.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved_energy(measurements, eng.evolved_time, env, M, model_params, site=site)
            logger.info(f"measurement took {(time.time()-t2):.3f}s")
            logger.info(f"t = {eng.evolved_time}")
            logger.info("---------------------------------")
        t_params['start_time'] = eng.evolved_time  # update clock for the following evolution

        # switch to other engine
        t_params['compression_method'] = 'variational'
        eng = ExpMPOEvolution(psi, M, t_params)
        if evolve_gs:
            eng_gs = ExpMPOEvolution(gs, M, t_params)


        if not os.path.exists("./energy_time_states/"):
            os.makedirs("./energy_time_states/")

        for i in range(tsteps_cont):
            t0 = time.time()
            eng.run()
            if evolve_gs:
                eng_gs.run()
            logger.info(f"time step took {(time.time()-t0):.3f}s")

            t2 = time.time()
            measure_evolved_energy(measurements, eng.evolved_time, env, M, model_params, site=site)
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
                    with h5py.File(f'./energy_time_states/{tsteps_init+previous_evolved_steps+i+1}'+f'time{round(eng.evolved_time, 2)}'+t_name, 'w') as f:
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

    # "./ground_states/GS_L33Cstylechi350_K-1.00Fx0.00Fy0.00Fz0.00W0.00EpFalse.h5" 
    gs_file = cmdargs[1]

    previous_run_file = None
    if run_time_restart:
        previous_run_file = './energy_time_states/3time0.03_Pert_Sm_Energy_chi350dt0.010ExpMPO_GS_L33Cstylechi350_K-1.00Fx0.00Fy0.00Fz0.00W0.00EpFalse.h5'

    op_type = "Sm"
    dt = 0.01
    t_method = "ExpMPO"
    tsteps_init = 1
    tsteps_cont = 1
    save_state = True
    evolve_gs = False
    chi_max = 300

    

    run_time(gs_file=gs_file, chi_max=chi_max, op_type=op_type, dt=dt, t_method=t_method, tsteps_cont=tsteps_cont, tsteps_init=tsteps_init,
             save_state=save_state, evolve_gs=evolve_gs, run_time_restart=run_time_restart, time_state_name=previous_run_file)
