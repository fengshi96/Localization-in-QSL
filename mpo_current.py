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








def current_op(lattice, site):
    """
    give the energy current density operator for a given site.
    
    Inputs:
    psi: any state, e.g. a time-dependent state exp(-iHt)|perturbed> modified in-place in mps environment
    lattice: Attribute lattice of the model, e.g. Honeycomb.lat
    site: list of sites in which available energy density operators are matched in terms of operators' site of reference

    Output:
    Energy current density operator

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

    # three-point terms xx+yy+zz
    ops_246_p = []
    ops_357_m = []
    ops_457_p = []
    ops_345_p = []
    ops_245_m = []
    ops_456_m = []

    # two-point terms x+y+z for A and B sublattices
    ops_35_p = []
    ops_24_m = []
    ops_57_p = []
    ops_46_m = []
    ops_24_p = []
    ops_46_p = []
    ops_57_m = []
    ops_35_m = []


    # energy current terms
    # We creat a mask that incompasses the largest support \Union(op_a, op_b)
    # such that the list of op_a matches that of op_b
    #                  2(0)             3(1)            4(2)            5(3)             6(4)            7(5)
    op_mask = [("Id", [-1], 0), ("Id", [-1], 1), ('Id', [0], 0), ('Id', [0], 1), ("Id", [1], 0), ("Id", [1], 1)]
    mps_inds = lattice.possible_multi_couplings(op_mask)[0]

    op_found = False
    for inds in mps_inds:
        if inds[0] in site:
            print(inds[0], "is in the site list")
            # use inds[0] as the site of reference for an operator
            op_found = True

            # three spin terms
            op_246_p = [('Sigmax', inds[0]), ('Sigmaz', inds[2]), ('Sigmay', inds[4])]  
            op_357_m = [('Sigmay', inds[1]), ('Sigmaz', inds[3]), ('Sigmax', inds[5])]  
            op_457_p = [('Sigmaz', inds[2]), ('Sigmay', inds[3]), ('Sigmax', inds[5])]  
            op_345_p = [('Sigmay', inds[1]), ('Sigmaz', inds[2]), ('Sigmax', inds[3])]  
            op_245_m = [('Sigmax', inds[0]), ('Sigmay', inds[2]), ('Sigmaz', inds[3])]  
            op_456_m = [('Sigmax', inds[2]), ('Sigmaz', inds[3]), ('Sigmay', inds[4])]  

            ops_246_p.append(op_246_p)
            ops_357_m.append(op_357_m)
            ops_457_p.append(op_457_p)
            ops_345_p.append(op_345_p)
            ops_245_m.append(op_245_m)
            ops_456_m.append(op_456_m)
            print("finished appending three-point strings")


            # two spin terms
            op_35_p = [('Sigmay', inds[1]), ('Sigmax', inds[3])]
            op_24_m = [('Sigmax', inds[0]), ('Sigmay', inds[2])]
            op_57_p = [('Sigmay', inds[3]), ('Sigmax', inds[5])]
            op_46_m = [('Sigmax', inds[2]), ('Sigmay', inds[4])]
            op_24_p = [('Sigmax', inds[0]), ('Sigmaz', inds[2])]
            op_46_p = [('Sigmaz', inds[2]), ('Sigmay', inds[4])]
            op_57_m = [('Sigmaz', inds[3]), ('Sigmax', inds[5])]
            op_35_m = [('Sigmay', inds[1]), ('Sigmaz', inds[3])]

            ops_35_p.append(op_35_p)
            ops_24_m.append(op_24_m)
            ops_57_p.append(op_57_p)
            ops_46_m.append(op_46_m)
            ops_24_p.append(op_24_p)
            ops_46_p.append(op_46_p)
            ops_57_m.append(op_57_m)
            ops_35_m.append(op_35_m)
            print("finished appending two-point strings")
            print("finished appending all operator strings")

    if not op_found:
        raise ValueError(site, "No available energy operator found according to the list of sites!")
    
    return ops_246_p, ops_357_m, ops_457_p, ops_345_p, ops_245_m, ops_456_m, \
           ops_35_p, ops_24_m, ops_57_p, ops_46_m, ops_24_p, ops_46_p, ops_57_m, ops_35_m



class CurrentOperators(CouplingMPOModel):
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
        h = Fz
        siteRef = model_params.get('siteRef', [0])
        print("siteRef: ", siteRef)

        Jx = J_K
        Jy = J_K
        Jz = J_K
        if J_K != -1:
            raise ValueError(f"Jx{Jx}, Jy{Jy}, Jz{Jz} must all be -1 (iso AFM) for the current version!")
        if not (Fx == Fy == Fz):
            raise ValueError(f"Fx{Fx}, Fy{Fy}, Fz{Fz} must be the same for the current version!")
        
        print("Jx, Jy, Jz: ", Jx, Jy, Jz)
        print("Fx, Fy, Fz: ", Fx, Fy, Fz)

        if model_params['order'] == 'default':
            print("Default order")

        ops_246_p, ops_357_m, ops_457_p, ops_345_p, ops_245_m, ops_456_m, \
        ops_53_p, ops_24_m, ops_75_p, ops_46_m, ops_24_p, ops_64_p, ops_75_m, ops_35_m = current_op(self.lat, siteRef)
        

        for  op_246_p, op_357_m, op_457_p, op_345_p, op_245_m, op_456_m, \
                op_53_p, op_24_m, op_75_p, op_46_m, op_24_p, op_64_p, op_75_m, op_35_m \
                in zip( ops_246_p, ops_357_m, ops_457_p, ops_345_p, ops_245_m, ops_456_m, \
                        ops_53_p, ops_24_m, ops_75_p, ops_46_m, ops_24_p, ops_64_p, ops_75_m, ops_35_m):
            
            # three-point terms add_multi_coupling_term
            self.add_multi_coupling_term(+2, [op_246_p[0][1], op_246_p[1][1], op_246_p[2][1]], [op_246_p[0][0], op_246_p[1][0], op_246_p[2][0]], ['Id', 'Id'])
            self.add_multi_coupling_term(-2, [op_357_m[0][1], op_357_m[1][1], op_357_m[2][1]], [op_357_m[0][0], op_357_m[1][0], op_357_m[2][0]], ['Id', 'Id'])
            self.add_multi_coupling_term(+1, [op_457_p[0][1], op_457_p[1][1], op_457_p[2][1]], [op_457_p[0][0], op_457_p[1][0], op_457_p[2][0]], ['Id', 'Id'])
            self.add_multi_coupling_term(+1, [op_345_p[0][1], op_345_p[1][1], op_345_p[2][1]], [op_345_p[0][0], op_345_p[1][0], op_345_p[2][0]], ['Id', 'Id'])
            self.add_multi_coupling_term(-1, [op_245_m[0][1], op_245_m[1][1], op_245_m[2][1]], [op_245_m[0][0], op_245_m[1][0], op_245_m[2][0]], ['Id', 'Id'])
            self.add_multi_coupling_term(-1, [op_456_m[0][1], op_456_m[1][1], op_456_m[2][1]], [op_456_m[0][0], op_456_m[1][0], op_456_m[2][0]], ['Id', 'Id'])
            print("op_246_p[0][1] (s1): ", op_246_p[0][1], "; op_246_p[1][1] (s2): ", op_246_p[1][1], "; op_246_p[2][1] (s3): ", op_246_p[2][1], "; op_246_p[0][0] (op1): ", op_246_p[0][0], "; op_246_p[1][0] (op2): ", op_246_p[1][0], "; op_246_p[2][0] (op3): ", op_246_p[2][0])
            print("op_357_m[0][1] (s1): ", op_357_m[0][1], "; op_357_m[1][1] (s2): ", op_357_m[1][1], "; op_357_m[2][1] (s3): ", op_357_m[2][1], "; op_357_m[0][0] (op1): ", op_357_m[0][0], "; op_357_m[1][0] (op2): ", op_357_m[1][0], "; op_357_m[2][0] (op3): ", op_357_m[2][0])
            print("op_457_p[0][1] (s1): ", op_457_p[0][1], "; op_457_p[1][1] (s2): ", op_457_p[1][1], "; op_457_p[2][1] (s3): ", op_457_p[2][1], "; op_457_p[0][0] (op1): ", op_457_p[0][0], "; op_457_p[1][0] (op2): ", op_457_p[1][0], "; op_457_p[2][0] (op3): ", op_457_p[2][0])
            print("op_345_p[0][1] (s1): ", op_345_p[0][1], "; op_345_p[1][1] (s2): ", op_345_p[1][1], "; op_345_p[2][1] (s3): ", op_345_p[2][1], "; op_345_p[0][0] (op1): ", op_345_p[0][0], "; op_345_p[1][0] (op2): ", op_345_p[1][0], "; op_345_p[2][0] (op3): ", op_345_p[2][0])
            print("op_245_m[0][1] (s1): ", op_245_m[0][1], "; op_245_m[1][1] (s2): ", op_245_m[1][1], "; op_245_m[2][1] (s3): ", op_245_m[2][1], "; op_245_m[0][0] (op1): ", op_245_m[0][0], "; op_245_m[1][0] (op2): ", op_245_m[1][0], "; op_245_m[2][0] (op3): ", op_245_m[2][0])
            print("op_456_m[0][1] (s1): ", op_456_m[0][1], "; op_456_m[1][1] (s2): ", op_456_m[1][1], "; op_456_m[2][1] (s3): ", op_456_m[2][1], "; op_456_m[0][0] (op1): ", op_456_m[0][0], "; op_456_m[1][0] (op2): ", op_456_m[1][0], "; op_456_m[2][0] (op3): ", op_456_m[2][0])
            print("finished adding three-spin coupling terms \n")

            # two-point terms
            self.add_coupling_term(+h, op_53_p[0][1], op_53_p[1][1], op_53_p[0][0], op_53_p[1][0])
            self.add_coupling_term(-h, op_24_m[0][1], op_24_m[1][1], op_24_m[0][0], op_24_m[1][0])
            self.add_coupling_term(+h, op_75_p[0][1], op_75_p[1][1], op_75_p[0][0], op_75_p[1][0])
            self.add_coupling_term(-h, op_46_m[0][1], op_46_m[1][1], op_46_m[0][0], op_46_m[1][0])
            self.add_coupling_term(+h, op_24_p[0][1], op_24_p[1][1], op_24_p[0][0], op_24_p[1][0])
            self.add_coupling_term(+h, op_64_p[0][1], op_64_p[1][1], op_64_p[0][0], op_64_p[1][0])
            self.add_coupling_term(-h, op_75_m[0][1], op_75_m[1][1], op_75_m[0][0], op_75_m[1][0])
            self.add_coupling_term(-h, op_35_m[0][1], op_35_m[1][1], op_35_m[0][0], op_35_m[1][0])
            print("op_53_p[0][1] (s1): ", op_53_p[0][1], "; op_53_p[1][1] (s2): ", op_53_p[1][1], "; op_53_p[0][0] (op1): ", op_53_p[0][0], "; op_53_p[1][0] (op2): ", op_53_p[1][0])
            print("op_24_m[0][1] (s1): ", op_24_m[0][1], "; op_24_m[1][1] (s2): ", op_24_m[1][1], "; op_24_m[0][0] (op1): ", op_24_m[0][0], "; op_24_m[1][0] (op2): ", op_24_m[1][0])
            print("op_75_p[0][1] (s1): ", op_75_p[0][1], "; op_75_p[1][1] (s2): ", op_75_p[1][1], "; op_75_p[0][0] (op1): ", op_75_p[0][0], "; op_75_p[1][0] (op2): ", op_75_p[1][0])
            print("op_46_m[0][1] (s1): ", op_46_m[0][1], "; op_46_m[1][1] (s2): ", op_46_m[1][1], "; op_46_m[0][0] (op1): ", op_46_m[0][0], "; op_46_m[1][0] (op2): ", op_46_m[1][0])
            print("op_24_p[0][1] (s1): ", op_24_p[0][1], "; op_24_p[1][1] (s2): ", op_24_p[1][1], "; op_24_p[0][0] (op1): ", op_24_p[0][0], "; op_24_p[1][0] (op2): ", op_24_p[1][0])
            print("op_64_p[0][1] (s1): ", op_64_p[0][1], "; op_64_p[1][1] (s2): ", op_64_p[1][1], "; op_64_p[0][0] (op1): ", op_64_p[0][0], "; op_64_p[1][0] (op2): ", op_64_p[1][0])
            print("op_75_m[0][1] (s1): ", op_75_m[0][1], "; op_75_m[1][1] (s2): ", op_75_m[1][1], "; op_75_m[0][0] (op1): ", op_75_m[0][0], "; op_75_m[1][0] (op2): ", op_75_m[1][0])
            print("op_35_m[0][1] (s1): ", op_35_m[0][1], "; op_35_m[1][1] (s2): ", op_35_m[1][1], "; op_35_m[0][0] (op1): ", op_35_m[0][0], "; op_35_m[1][0] (op2): ", op_35_m[1][0])
            print("finished adding two-spin coupling terms \n")

        print("finished adding all coupling terms \n")
 








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
    op_params['siteRef'] = [6]
    print("order: ", order)
    print(op_params.keys())

    # M = Kitaev_Ladder(state['model_params'])
    E_MPO = CurrentOperators(op_params).H_MPO 
    print(E_MPO.variance(psi))
    print(E_MPO.expectation_value(gs))



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
