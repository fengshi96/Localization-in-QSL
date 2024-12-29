"""
implement extended Kitaev on a honeycomb lattice
Sign Convention:
H = (-Jx XX - Jy YY - Jz ZZ) - Gamma (...) - Gamma_Prime(...) - J_H(...) - J_H3(...) - W Wp
"""
from tenpy.networks.site import SpinHalfSite
# from tenpy.linalg import np_conserved as npc
from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Honeycomb
import math



class Kitaev_Extended(CouplingMPOModel):
    r"""
    Kitaev Hamiltonian with a hole,

    convention such that SzSz bonds are horizontal

    """
    def init_lattice(self, model_params):
        Lx = model_params.get('Lx', 0.)
        Ly = model_params.get('Ly', 0.)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = model_params.get('bc', ['open', 'periodic'])
        order = model_params.get('order', 'default')
        site = SpinHalfSite(conserve='None')
        lat = Honeycomb(Lx, Ly, site, bc=bc, bc_MPS=bc_MPS, order=order)
        return lat

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_terms(self, model_params):
        # See https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.033011 (Eq. 2)
        # 0) read out/set default parameters
        J_K = model_params.get('J_K', 1.) # isotropic Kitaev coupling
        Fx = model_params.get('Fx', 0.)  # magnetic field in x
        Fy = model_params.get('Fy', 0.)  # magnetic field in y
        Fz = model_params.get('Fz', 0.)  # magnetic field in z
        W = model_params.get('W', 0.)


        # allow for anisotropic couplings
        Jx = J_K
        Jy = J_K
        Jz = J_K

        # Kitaev interactions
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            # Kitaev terms
            if list(dx) == [1, 0]: # ZZ bonds
                self.add_coupling(-Jz, u1, 'Sigmaz', u2, 'Sigmaz', dx)
            elif list(dx) == [0, 1]: # XX bonds
                self.add_coupling(-Jx, u1, 'Sigmax', u2, 'Sigmax', dx)
            elif list(dx) ==[0, 0]: # YY bonds
                self.add_coupling(-Jy, u1, 'Sigmay', u2, 'Sigmay', dx)
            else:
                raise ValueError('wrong bonds?')

        # magnetic fields:
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(Fx / math.sqrt(2), u, 'Sigmax')
            self.add_onsite(Fy / math.sqrt(2), u, 'Sigmay')
            self.add_onsite(Fz / math.sqrt(2), u, 'Sigmaz')


        # add Wp plaquette terms to help DMRG
        ops = [('Sigmaz', [0, 0], 0), ('Sigmax', [0, 0], 1), ('Sigmay', [1, 0], 0), ('Sigmaz', [1, -1], 1), ('Sigmax', [1, -1], 0), ('Sigmay', [0, -1], 1)]
        self.add_multi_coupling(-W, ops)


    def draw(self, ax, coupling=None, wrap=False):
        # for drawing lattice with MPS indices
        self.lat.plot_coupling(ax, coupling=None, wrap=False)
        self.lat.plot_basis(ax, origin=(-0.0, -0.0), shade=None)
        # print(self.lat.bc)
        self.lat.plot_bc_identified(ax, direction=-1, origin=None, cylinder_axis=True)
        self.lat.plot_order(ax)

    def positions(self):
        # return space positions of all sites
        return self.lat.mps2lat_idx(range(self.lat.N_sites))




# ----------------------------------------------------------
# ----------------- Debug Kitaev_Extended() ----------------
# ----------------------------------------------------------
def test(**kwargs):
    print("test module")
    Lx = kwargs['Lx']
    Ly = kwargs['Ly']
    order = kwargs.get('order', 'default')
    J_K = kwargs['J_K']
    Jx = kwargs['Jx']
    Jy = kwargs['Jy']
    Jz = kwargs['Jz']
    J_H = kwargs['J_H']
    Jnnn = kwargs['Jnnn']
    G = kwargs['G']
    G_P = kwargs['G_P']
    W = kwargs['W']
    # bc = 'open'

    model_params = dict(Lx=Lx, Ly=Ly, order=order,
                        J_K=J_K, J_H=J_H, Jnnn=Jnnn, G=G, G_P=G_P, W=W,
                        Jx=Jx, Jy=Jy, Jz=Jz)
    M = Kitaev_Extended(model_params)
    
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

    Lx = 6
    Ly = 3
    J_K = 1.0
    Jx = 0.5
    Jy = 0.5
    Jz = 0.5
    J_H = 0
    Jnnn = 0
    G = 0
    G_P = 0
    W = 0
    K = 0
    Fx = 0
    Fy = 0
    Fz = 0
    order = 'Cstyle' 
    bc = ['open', 'periodic']
    test(Lx=Lx, Ly=Ly, order=order, bc=bc,
                        J_K=J_K, J_H=J_H, Jnnn=Jnnn, G=G, G_P=G_P, Fx=Fx, Fy=Fy, Fz=Fz, W=W,
                        Jx=Jx, Jy=Jy, Jz=Jz)
