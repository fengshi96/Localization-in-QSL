"""
implement extended Kitaev on a ladder lattice
Sign Convention:
H = (-Jx XX - Jy YY - Jz ZZ) - Fx Sx - Fy Sy - Fz Sz
"""
from tenpy.networks.site import SpinHalfSite
# from tenpy.linalg import np_conserved as npc
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Honeycomb
from tenpy.models.lattice import Ladder
from energy_dynamics_ladder import measure_energy_densities
import math








def energy_op(lattice, site):
    """
    give the energy density operators for a given site.
    
    Inputs:
    psi: any state, e.g. a time-dependent state exp(-iHt)|perturbed> modified in-place in mps environment
    lattice: Attribute lattice of the model, e.g. Honeycomb.lat
    site: list of sites in which available energy density operators are matched in terms of operators' site of reference

    Output:
    energy density operators for the given site

    The lattice is labeled as follows (e.g. for a Lx = 8 ladder under OBC):
    1 - x - 3 - y - 5 - x - 7 - y - 9 - x - 11 - y - 13 - x - 15 - y - 17 - x - 19 - y - 21 - x - 23
    |       |       |       |       |       |        |        |        |        |        |        |
    z       z       z       z       z       z        z        z        z        z        z        z
    |       |       |       |       |       |        |        |        |        |        |        |
    0 - y - 2 - x - 4 - y - 6 - x - 8 - y - 10 - x - 12 - y - 14 - x - 16 - y - 18 - x - 20 - y - 22 
    
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
    
    return ops_lz, ops_lx, ops_ly, ops_rz, ops_rx, ops_ry, ops_mz, ops_Ax, ops_Ay, ops_Az, ops_Bx, ops_By, ops_Bz, ops_lAx, ops_lAy, ops_lAz, ops_lBx, ops_lBy, ops_lBz, ops_rAx, ops_rAy, ops_rAz, ops_rBx, ops_rBy, ops_rBz




class EnergyOperators(CouplingMPOModel):
    r"""

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
    def init_lattice(self, model_params):
        Lx = model_params.get('Lx', 0.)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = model_params.get('bc', 'open')
        order = model_params.get('order', 'default')
        site = SpinHalfSite(conserve='None')
        lat = Ladder(Lx, site, bc=bc, bc_MPS=bc_MPS, order='default')
        return lat

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def lattice_folded(self, Lx, bc):
        site = SpinHalfSite(conserve='None')
        lat_defalut_order = Ladder(Lx, site, bc=bc, bc_MPS='finite', order='folded')
        return lat_defalut_order

    def init_terms(self, model_params):
        # See https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.033011 (Eq. 2)
        # 0) read out/set default parameters
        J_K = model_params.get('J_K', -1.) # isotropic Kitaev coupling
        Fx = model_params.get('Fx', 0.)  # magnetic field in x
        Fy = model_params.get('Fy', 0.)  # magnetic field in y
        Fz = model_params.get('Fz', 0.)  # magnetic field in z
        siteRef = model_params.get('siteRef', [0])
        print("siteRef: ", siteRef)


        # allow for anisotropic couplings
        Jx = J_K
        Jy = J_K
        Jz = J_K
        print("Jx, Jy, Jz: ", Jx, Jy, Jz)
        print("Fx, Fy, Fz: ", Fx, Fy, Fz)

        if model_params['order'] == 'default':
            print("Default order")

        ops_lz, ops_lx, ops_ly, ops_rz, ops_rx, ops_ry, ops_mz, ops_Ax, \
            ops_Ay, ops_Az, ops_Bx, ops_By, ops_Bz, ops_lAx, ops_lAy, \
                ops_lAz, ops_lBx, ops_lBy, ops_lBz, ops_rAx, ops_rAy, ops_rAz, \
                    ops_rBx, ops_rBy, ops_rBz = energy_op(self.lat, siteRef)
        

        for op_lz, op_lx, op_ly, op_rz, op_rx, op_ry, op_mz, op_Ax, op_Ay, op_Az, op_Bx, op_By, op_Bz, \
             op_lAx, op_lAy, op_lAz, op_lBx, op_lBy, op_lBz, op_rAx, op_rAy, op_rAz, op_rBx, op_rBy, op_rBz \
                in zip(ops_lz, ops_lx, ops_ly, ops_rz, ops_rx, ops_ry, ops_mz, ops_Ax, ops_Ay, ops_Az, ops_Bx, ops_By, ops_Bz, \
                       ops_lAx, ops_lAy, ops_lAz, ops_lBx, ops_lBy, ops_lBz, ops_rAx, ops_rAy, ops_rAz, ops_rBx, ops_rBy, ops_rBz):
            
            # op1, op2, s1, s2 = op_lz[0][0], op_lz[1][0], op_lz[0][1], op_lz[1][1]

            # two-point terms
            self.add_coupling_term(-Jz, op_lz[0][1], op_lz[1][1], op_lz[0][0], op_lz[1][0])
            self.add_coupling_term(-Jx, op_lx[0][1], op_lx[1][1], op_lx[0][0], op_lx[1][0])
            self.add_coupling_term(-Jy, op_ly[0][1], op_ly[1][1], op_ly[0][0], op_ly[1][0])
            self.add_coupling_term(-Jz, op_rz[0][1], op_rz[1][1], op_rz[0][0], op_rz[1][0])
            self.add_coupling_term(-Jx, op_rx[0][1], op_rx[1][1], op_rx[0][0], op_rx[1][0])
            self.add_coupling_term(-Jy, op_ry[0][1], op_ry[1][1], op_ry[0][0], op_ry[1][0])
            self.add_coupling_term(-Jz, op_mz[0][1], op_mz[1][1], op_mz[0][0], op_mz[1][0])

            print("op_lz[0][1] (s1): ", op_lz[0][1], "; op_lz[1][1] (s2): ", op_lz[1][1], "; op_lz[0][0] (op1): ", op_lz[0][0], "; op_lz[1][0] (op2): ", op_lz[1][0])
            print("op_lx[0][1] (s1): ", op_lx[0][1], "; op_lx[1][1] (s2): ", op_lx[1][1], "; op_lx[0][0] (op1): ", op_lx[0][0], "; op_lx[1][0] (op2): ", op_lx[1][0])
            print("op_ly[0][1] (s1): ", op_ly[0][1], "; op_ly[1][1] (s2): ", op_ly[1][1], "; op_ly[0][0] (op1): ", op_ly[0][0], "; op_ly[1][0] (op2): ", op_ly[1][0])
            print("op_rz[0][1] (s1): ", op_rz[0][1], "; op_rz[1][1] (s2): ", op_rz[1][1], "; op_rz[0][0] (op1): ", op_rz[0][0], "; op_rz[1][0] (op2): ", op_rz[1][0])
            print("op_rx[0][1] (s1): ", op_rx[0][1], "; op_rx[1][1] (s2): ", op_rx[1][1], "; op_rx[0][0] (op1): ", op_rx[0][0], "; op_rx[1][0] (op2): ", op_rx[1][0])
            print("op_ry[0][1] (s1): ", op_ry[0][1], "; op_ry[1][1] (s2): ", op_ry[1][1], "; op_ry[0][0] (op1): ", op_ry[0][0], "; op_ry[1][0] (op2): ", op_ry[1][0])
            print("op_mz[0][1] (s1): ", op_mz[0][1], "; op_mz[1][1] (s2): ", op_mz[1][1], "; op_mz[0][0] (op1): ", op_mz[0][0], "; op_mz[1][0] (op2): ", op_mz[1][0])
            print("finished adding coupling terms \n")

            # one-point terms
            self.add_onsite_term(Fx, op_Ax[0][1], op_Ax[0][0])
            self.add_onsite_term(Fy, op_Ay[0][1], op_Ay[0][0])
            self.add_onsite_term(Fz, op_Az[0][1], op_Az[0][0])
            self.add_onsite_term(Fx, op_Bx[0][1], op_Bx[0][0])
            self.add_onsite_term(Fy, op_By[0][1], op_By[0][0])
            self.add_onsite_term(Fz, op_Bz[0][1], op_Bz[0][0])
            self.add_onsite_term(Fx, op_lAx[0][1], op_lAx[0][0])
            self.add_onsite_term(Fy, op_lAy[0][1], op_lAy[0][0])
            self.add_onsite_term(Fz, op_lAz[0][1], op_lAz[0][0])
            self.add_onsite_term(Fx, op_lBx[0][1], op_lBx[0][0])
            self.add_onsite_term(Fy, op_lBy[0][1], op_lBy[0][0])
            self.add_onsite_term(Fz, op_lBz[0][1], op_lBz[0][0])
            self.add_onsite_term(Fx, op_rAx[0][1], op_rAx[0][0])
            self.add_onsite_term(Fy, op_rAy[0][1], op_rAy[0][0])
            self.add_onsite_term(Fz, op_rAz[0][1], op_rAz[0][0])
            self.add_onsite_term(Fx, op_rBx[0][1], op_rBx[0][0])
            self.add_onsite_term(Fy, op_rBy[0][1], op_rBy[0][0])
            self.add_onsite_term(Fz, op_rBz[0][1], op_rBz[0][0])

            print("op_Ax[0][1] (s1): ", op_Ax[0][1], "; op_Ax[0][0] (op1): ", op_Ax[0][0])
            print("op_Ay[0][1] (s1): ", op_Ay[0][1], "; op_Ay[0][0] (op1): ", op_Ay[0][0])
            print("op_Az[0][1] (s1): ", op_Az[0][1], "; op_Az[0][0] (op1): ", op_Az[0][0])
            print("op_Bx[0][1] (s1): ", op_Bx[0][1], "; op_Bx[0][0] (op1): ", op_Bx[0][0])
            print("op_By[0][1] (s1): ", op_By[0][1], "; op_By[0][0] (op1): ", op_By[0][0])
            print("op_Bz[0][1] (s1): ", op_Bz[0][1], "; op_Bz[0][0] (op1): ", op_Bz[0][0])
            print("op_lAx[0][1] (s1): ", op_lAx[0][1], "; op_lAx[0][0] (op1): ", op_lAx[0][0])
            print("op_lAy[0][1] (s1): ", op_lAy[0][1], "; op_lAy[0][0] (op1): ", op_lAy[0][0])
            print("op_lAz[0][1] (s1): ", op_lAz[0][1], "; op_lAz[0][0] (op1): ", op_lAz[0][0])
            print("op_lBx[0][1] (s1): ", op_lBx[0][1], "; op_lBx[0][0] (op1): ", op_lBx[0][0])
            print("op_lBy[0][1] (s1): ", op_lBy[0][1], "; op_lBy[0][0] (op1): ", op_lBy[0][0])
            print("op_lBz[0][1] (s1): ", op_lBz[0][1], "; op_lBz[0][0] (op1): ", op_lBz[0][0])
            print("op_rAx[0][1] (s1): ", op_rAx[0][1], "; op_rAx[0][0] (op1): ", op_rAx[0][0])
            print("op_rAy[0][1] (s1): ", op_rAy[0][1], "; op_rAy[0][0] (op1): ", op_rAy[0][0])
            print("op_rAz[0][1] (s1): ", op_rAz[0][1], "; op_rAz[0][0] (op1): ", op_rAz[0][0])
            print("op_rBx[0][1] (s1): ", op_rBx[0][1], "; op_rBx[0][0] (op1): ", op_rBx[0][0])
            print("op_rBy[0][1] (s1): ", op_rBy[0][1], "; op_rBy[0][0] (op1): ", op_rBy[0][0])
            print("op_rBz[0][1] (s1): ", op_rBz[0][1], "; op_rBz[0][0] (op1): ", op_rBz[0][0])
            print("finished adding operators \n")






class Onsite(CouplingMPOModel):
    def init_lattice(self, model_params):
        Lx = model_params.get('Lx', 0.)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = model_params.get('bc', 'open')
        order = model_params.get('order', 'default')
        site = SpinHalfSite(conserve='None')
        lat = Ladder(Lx, site, bc=bc, bc_MPS=bc_MPS, order='default')
        return lat

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def lattice_folded(self, Lx, bc):
        site = SpinHalfSite(conserve='None')
        lat_defalut_order = Ladder(Lx, site, bc=bc, bc_MPS='finite', order='folded')
        return lat_defalut_order

    def init_terms(self, model_params):
        siteRef = model_params.get('siteRef', 0)
        opRef = model_params.get('opRef', 'Sigmaz')
        print("siteRef: ", siteRef)

        self.add_onsite_term(1, siteRef, opRef)


def purturbation_energy(psi, op_params, option):
    E_MPO = EnergyOperators(op_params).H_MPO 
    E_MPO.apply(psi, option)


# ----------------------------------------------------------
# ----------------- Debug Kitaev_Extended() ----------------
# ----------------------------------------------------------
def test(**kwargs):
    print("test module")
    
    # load ground state
    gs_file = "./ground_states/GS_L11defaultchi90_K-1.00Fx-0.70Fy-0.70Fz-0.70.h5" 

    with h5py.File(gs_file, 'r') as f:
        state = hdf5_io.load_from_hdf5(f)

    psi = state['gs'].copy()
    gs = state['gs'].copy()

    op_params = state['model_params']
    del op_params['dmrg_restart']
    op_params['siteRef'] = [8]
    print("order: ", order)
    print(op_params.keys())

    # M = Kitaev_Ladder(state['model_params'])
    E_MPO = EnergyOperators(op_params).H_MPO 

    onsite_params = op_params.copy()
    onsite_params['siteRef'] = 10
    onsite_params['opRef'] = 'Sigmax'
    Op = Onsite(onsite_params).H_MPO

    chi_max = 90
    option = {
            'compression_method': 'zip_up', 'm_temp': 2, 'trunc_weight': 0.5,
            'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10, 'trunc_cut': 1e-10 }
            }
    # E_MPO.apply(psi, option)
    Op.apply(gs, option)
    Op.apply(psi, option)
    E_MPO_EXP = E_MPO.expectation_value(gs)
    E_MPO_VAR = E_MPO.variance(gs)

    purturbation_energy(psi, op_params, option)

    Overlap = gs.overlap(psi) 

    purturbation_energy(psi, op_params, option)
    Overlap2 = gs.overlap(psi)

    print("Overlap <h>: ", Overlap)
    print("Overlap <h2>: ", Overlap2)

    print("Relative Variance: ", E_MPO_EXP, E_MPO_VAR, (Overlap2 - Overlap ** 2))




    # site = range(0, M.lat.N_sites - 2, 4)
    # energies0 = measure_energy_densities(gs, M.lat, state['model_params'], site)
    # energies1 = measure_energy_densities(psi, M.lat, state['model_params'], site)
    # print("energies_gs: ", energies0, "total_energy: ", np.sum(energies0))
    # print("energies_pert: ", energies1, "total_energy: ", np.sum(energies1))




if __name__ == "__main__":
    import numpy as np
    import sys, os
    import h5py
    from tenpy.tools import hdf5_io
    from tenpy.networks.mps import MPS, MPSEnvironment
    from tenpy.networks.mpo import MPOEnvironment
    from model_ladder import Kitaev_Ladder
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    # print(cmdargs[1])

    J_K = -1.0
    Fx = 0
    Fy = 0
    Fz = 0
    order = 'default'  
    bc = 'open'

    Lx = 11
    if (bc == 'periodic' and (Lx / 2) % 2 != 1) or (bc == 'periodic' and Lx - int(Lx)!=0):  ## FIX ME!
        raise ValueError("Lx must be even and we need odd number of unit cells along the ladder to tile the energy operator!")

    test(Lx=Lx, order=order, bc=bc, J_K=J_K, Fx=Fx, Fy=Fy, Fz=Fz)
