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




def fluxes_op(lattice, site='all'):
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

    # two-point terms xx+yy+zz
    ops_s1 = []
    ops_s2 = []
    ops_wp = []

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
    
    return ops_s1, ops_s2, ops_wp





class S1Operators(CouplingMPOModel):
    r"""

    The lattice is labeled as follows (e.g. for a Lx = 8 ladder under OBC):
    1 - x - 3 - y - 5 - x - 7 - y - 9 - x - 11 - y - 13 - x - 15
    |       |       |       |       |       |        |        |
    z   s1  z   s2  z       z       z       z        z        z
    |       |       |       |       |       |        |        |
    0 - y - 2 - x - 4 - y - 6 - x - 8 - y - 10 - x - 12 - y - 14
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
        siteRef = model_params.get('siteRef', [0])
        print("[S1Operators] siteRef: ", siteRef)

        ops_s1, ops_s2, ops_wp = fluxes_op(self.lat, siteRef)

        for op_s1, _, _ in zip(ops_s1, ops_s2, ops_wp):
            self.add_multi_coupling_term(1, [op_s1[0][1], op_s1[1][1], op_s1[2][1], op_s1[3][1]], [op_s1[0][0], op_s1[1][0], op_s1[2][0], op_s1[3][0]], ['Id', 'Id', 'Id'])



class S2Operators(CouplingMPOModel):
    r"""

    The lattice is labeled as follows (e.g. for a Lx = 8 ladder under OBC):
    1 - x - 3 - y - 5 - x - 7 - y - 9 - x - 11 - y - 13 - x - 15
    |       |       |       |       |       |        |        |
    z   s1  z   s2  z       z       z       z        z        z
    |       |       |       |       |       |        |        |
    0 - y - 2 - x - 4 - y - 6 - x - 8 - y - 10 - x - 12 - y - 14
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
        siteRef = model_params.get('siteRef', [0])
        print("[S2Operators] siteRef: ", siteRef)

        ops_s1, ops_s2, ops_wp = fluxes_op(self.lat, siteRef)

        for _, op_s2, _ in zip(ops_s1, ops_s2, ops_wp):
            self.add_multi_coupling_term(1, [op_s2[0][1], op_s2[1][1], op_s2[2][1], op_s2[3][1]], [op_s2[0][0], op_s2[1][0], op_s2[2][0], op_s2[3][0]], ['Id', 'Id', 'Id'])




def purturb_S1(psi, op_params, option):
    E_MPO = S1Operators(op_params).H_MPO 
    E_MPO.apply(psi, option)


def purturb_S2(psi, op_params, option):
    E_MPO = S2Operators(op_params).H_MPO 
    E_MPO.apply(psi, option)


# ----------------------------------------------------------
# ----------------- Debug Kitaev_Extended() ----------------
# ----------------------------------------------------------
def test(**kwargs):
    print("test module")
    
    # load ground state
    gs_file = "./ground_states/GS_L11defaultchi90_K-1.00Fx-0.00Fy-0.00Fz-0.00.h5" 

    with h5py.File(gs_file, 'r') as f:
        state = hdf5_io.load_from_hdf5(f)

    psi = state['gs'].copy()
    gs = state['gs'].copy()

    op_params = state['model_params']
    del op_params['dmrg_restart']
    op_params['siteRef'] = [4]
    print("order: ", order)
    print(op_params.keys())

    # M = Kitaev_Ladder(state['model_params'])
    E_MPO = S1Operators(op_params).H_MPO 
    S1_exp = E_MPO.expectation_value(gs)
    

    E_MPO = S2Operators(op_params).H_MPO
    S2_exp = E_MPO.expectation_value(gs)
    
    print("S1_exp: ", S1_exp)
    print("S2_exp: ", S2_exp)









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
