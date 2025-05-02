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
import math



class Kitaev_Ladder(CouplingMPOModel):
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
        J_K = model_params.get('J_K', 1.) # isotropic Kitaev coupling
        Fx = model_params.get('Fx', 0.)  # magnetic field in x
        Fy = model_params.get('Fy', 0.)  # magnetic field in y
        Fz = model_params.get('Fz', 0.)  # magnetic field in z


        # allow for anisotropic couplings
        Jx = J_K
        Jy = J_K
        Jz = J_K

        # # Kitaev interactions
        if model_params['order'] == 'default':
            print("Default order")

        couplingsX_latindx = []
        couplingsY_latindx = []
        couplingsZ_latindx = []

        for i in range(self.lat.N_sites):
            if i % 2 == 0 and i < self.lat.N_sites - 1:
                if model_params['order'] == 'default':
                    print("ZZ coupling between ", i, " and ", i+1)
                    self.add_coupling_term(-Jz, i, i+1, 'Sigmaz', 'Sigmaz')
                couplingsZ_latindx.append([self.lat.mps2lat_idx(i), self.lat.mps2lat_idx(i+1)])
            if (i % 4 == 0 or (i - 3) % 4 == 0) and i < self.lat.N_sites - 2:
                if model_params['order'] == 'default':
                    print("YY coupling between ", i, " and ", i+2)
                    self.add_coupling_term(-Jy, i, i+2, 'Sigmay', 'Sigmay')
                couplingsY_latindx.append([self.lat.mps2lat_idx(i), self.lat.mps2lat_idx(i+2)])
            if ((i - 1) % 4 == 0 or (i - 2) % 4 == 0) and i < self.lat.N_sites - 2:
                if model_params['order'] == 'default':
                    print("XX coupling between ", i, " and ", i+2)
                    self.add_coupling_term(-Jx, i, i+2, 'Sigmax', 'Sigmax')
                couplingsX_latindx.append([self.lat.mps2lat_idx(i), self.lat.mps2lat_idx(i+2)])

        if model_params['bc'] == 'periodic':
            if model_params['order'] == 'default':
                self.add_coupling_term(-Jx, 0, self.lat.N_sites - 2, 'Sigmax', 'Sigmax')
                self.add_coupling_term(-Jy, 1, self.lat.N_sites - 1, 'Sigmay', 'Sigmay')
                print("XX coupling between ", 0, " and ", self.lat.N_sites - 2)
                print("YY coupling between ", 1, " and ", self.lat.N_sites - 1)
            couplingsX_latindx.append([self.lat.mps2lat_idx(0), self.lat.mps2lat_idx(self.lat.N_sites - 2)])
            couplingsY_latindx.append([self.lat.mps2lat_idx(1), self.lat.mps2lat_idx(self.lat.N_sites - 1)])


        if model_params['order'] == 'folded': 
            print("Folded order \n")

            lat_folded = self.lattice_folded(model_params['Lx'], model_params['bc'])
            # we update the lattice on-the-fly to folded order; this is not the most elegant solution, but it works
            self.lat = lat_folded

            for item in couplingsX_latindx:
                coupling_unsorted = lat_folded.lat2mps_idx(item)
                coupling = [0, 0]
                coupling[0] = min(coupling_unsorted)
                coupling[1] = max(coupling_unsorted)
                
                print("add XX coupling in folded lattice: ", coupling)
                self.add_coupling_term(-Jx, coupling[0], coupling[1], 'Sigmax', 'Sigmax')

            print('\n')
            for item in couplingsY_latindx:
                coupling_unsorted = lat_folded.lat2mps_idx(item)
                coupling = [0, 0]
                coupling[0] = min(coupling_unsorted)
                coupling[1] = max(coupling_unsorted)
                print("add YY coupling in folded lattice: ", coupling)
                self.add_coupling_term(-Jy, coupling[0], coupling[1], 'Sigmay', 'Sigmay')
            
            print('\n')
            for i in range(self.lat.N_sites):
                if i % 2 == 0 and i < self.lat.N_sites - 1:
                    print("add ZZ coupling in folded lattice: ", [i, i+1])
                    self.add_coupling_term(-Jz, i, i+1, 'Sigmaz', 'Sigmaz')
            

        # magnetic fields:
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(Fx, u, 'Sigmax')
            self.add_onsite(Fy, u, 'Sigmay')
            self.add_onsite(Fz, u, 'Sigmaz')



    def draw(self, ax, coupling=None, wrap=False):
        # for drawing lattice with MPS indices
        self.lat.plot_coupling(ax, coupling=None, wrap=False)
        self.lat.plot_basis(ax, origin=(-0.0, -0.0), shade=None)
        # print(self.lat.bc)
        # self.lat.plot_bc_identified(ax, direction=-1, origin=None, cylinder_axis=True)
        self.lat.plot_order(ax, textkwargs={'color': 'blue', 'fontsize': 4})

    def positions(self):
        # return space positions of all sites
        return self.lat.mps2lat_idx(range(self.lat.N_sites))




# ----------------------------------------------------------
# ----------------- Debug Kitaev_Extended() ----------------
# ----------------------------------------------------------
def test(**kwargs):
    print("test module")
    Lx = kwargs['Lx']
    order = kwargs.get('order', 'default')
    J_K = kwargs['J_K']
    bc = kwargs['bc']

    model_params = dict(Lx=Lx, order=order, bc=bc,
                        J_K=J_K, Fx=kwargs['Fx'], Fy=kwargs['Fy'], Fz=kwargs['Fz'])
    M = Kitaev_Ladder(model_params)
    
    pos = M.positions()
    print("pos:\n", pos, "\n")
    
    test = np.array([(str(i)) for i in pos])
    retest = test.reshape(model_params['Lx'],-1).T
    print(retest,'\n')
    print(retest[::2,:])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1,  figsize=(8,6))     
    M.draw(ax)
    plt.savefig("./figure.pdf", dpi=300, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    import numpy as np
    import sys
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    # print(cmdargs[1])

    J_K = 1.0
    Fx = 1
    Fy = 1
    Fz = 1
    order = 'default'  
    bc = 'open'

    Lx = 43
    if (bc == 'periodic' and (Lx / 2) % 2 != 1) or (bc == 'periodic' and Lx - int(Lx)!=0):  ## FIX ME!
        raise ValueError("Lx must be even and we need odd number of unit cells along the ladder to tile the energy operator!")

    test(Lx=Lx, order=order, bc=bc, J_K=J_K, Fx=Fx, Fy=Fy, Fz=Fz)
