import numpy as np
import scipy as sp
import os
import h5py
import math
from tenpy.tools import hdf5_io
import matplotlib
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = "serif"

def lpc(signal, order):
    """Compute the Linear Prediction Coefficients.

    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:

      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]

    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.

    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)
    """
    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a lenght >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, signal.dtype)
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, 'full')
        r[:nx] = x[signal.size-1:signal.size+order]
        phi = sp.linalg.solve_toeplitz(r[:-1], r[1:])
        return np.concatenate(([1.], phi))
    else:
        return np.ones(1, dtype=signal.dtype)

def linear_prediction(signal, N, M=None, unbiased=False):
    S = signal.shape[0]
    if M is None:
        M = S//2
    M = min(M, S)

    # prediction
    pred = np.zeros(S + N, dtype=signal.dtype)
    pred[:S] = signal[:]

    # remove bias towards zero
    if unbiased:
        mean = np.mean(pred[:S])
    else:
        mean = 0
    pred -= mean

    # coefficients
    d = lpc(pred[:S], M)[1:]

    for j in range(N):
        pred[S+j] = np.dot(d, pred[S-M+j: S+j][::-1])

    return pred + mean

def Cij_matrixForm(Cij_vector, times_steps, Ly, Lx):
    Cij = np.zeros(((times_steps, Ly*2, Lx)), dtype=complex)
    for t_indx, time in enumerate(time_keys):
        cij = np.array((measurements[time]['Cij_Sz']))
        cij = cij.reshape(model_params['Lx'],-1).T
        Cij[t_indx, :, :] = cij


def C_FT(Cij, x_init=None, y_init=0):
    """ Fourier trafo for spectrum """
    if x_init is None:
        x_init = Cij.shape[1]//2

    kxlist = np.fft.fftshift(np.fft.fftfreq(Cij.shape[1]))*2*np.pi
    if Cij.shape[1] % 2 == 0:
        kxlist = np.append(kxlist, -kxlist[0])
        
    kylist = np.fft.fftshift(np.fft.fftfreq(Cij.shape[2]))*2*np.pi
    if Cij.shape[2] % 2 == 0:
        kylist = np.append(kylist, -kylist[0])

    # Fourier trafo in x direction
    Cky = np.fft.fft(np.roll(Cij, -x_init, axis=1), axis=1)
    Cky = np.fft.fftshift(Cky, axes=1)
    if Cij.shape[1] % 2 == 0:
        Cky = np.append(Cky, Cky[:,0,:].reshape(Cky.shape[0], 1, Cky.shape[2]), axis=1)

    # Fourier trafo in y direction
    Ck = np.fft.fft(np.roll(Cky, -y_init, axis=2), axis=2)
    Ck = np.fft.fftshift(Ck, axes=2)
    if Cij.shape[2] % 2 == 0:
        Ck = np.append(Ck, Ck[:,:,0].reshape(Ck.shape[0], Ck.shape[1], 1), axis=2)

    return Ck, kxlist, kylist


def tempFT(Ck, wmin, wmax, dt_corr):
    """Fourier trafo in time
    Inputs:
    Ck: Original time series data, shape (N, kx_dim, ky_dim), where N is the number of time steps.
    dt_corr: Time step size between successive measurements in the correlation function data.
    """
    # Ck[::-1]: Reverses Ck along the time axis.
    # Ck[::-1][:-1]: Excludes the last element to avoid duplicating the time t=0 (assuming t=0 is at index 0).
    # Ck[::-1][:-1].conj(): Takes the complex conjugate of the reversed Ck, ensuring Hermitian symmetry in time.
    # Ctk: An array of shape (2N - 1, kx_dim, ky_dim), representing the extended time series from negative to positive times.
    Ctk = np.vstack([Ck, Ck[::-1][:-1].conj()])

    # num_w is the total number of frequency points corresponding to the extended time series.
    num_w = Ck.shape[0]*2-1
    wlist = np.fft.fftfreq(num_w, d=dt_corr)*2*np.pi

    wlist = np.fft.fftshift(wlist)
    Akw = np.fft.fftshift(np.fft.ifft(Ctk*num_w, axis=0), axes=0) #FFT along time axis-0

    factor = len(wlist)*(wlist[1]-wlist[0])
    Akw /= factor
    w0 = np.argmin(np.abs(wlist-wmin))
    w1 = np.argmin(np.abs(wlist-wmax))
    wlist = wlist[w0:w1]
    Akw = Akw[w0:w1]

    return Akw, wlist


def make_spectrum(Cij, delta_t, gauss_sigma, n_lp, wmin, wmax, sublattice='A', sub_only=False):
    """
    Inputs:
    Cij: real space correlation functions at different time
         axis-0: time
         axis-1: x coord
         axis-2: y coord
    delta_t: time step from the time evolution
    gauss_sigma: width of gaussion filter
    n_lp: factor for linear prediction, e.g. for n_lp = 2 will append twice as many time step as the initial time array
    wmin: lower energy
    wmax: maximal energy
    sublattice: in which sublattice is the hole
    sub_only: if True only return spectrum for one sublattice
    ----------------------------------------------------------
    The suffix of the Cij array at a time slice is of the following form
    [
        ['[0 0 0]' '[1 0 0]' '[2 0 0]' '[3 0 0]']
        ['[0 0 1]' '[1 0 1]' '[2 0 1]' '[3 0 1]']
        ['[0 1 0]' '[1 1 0]' '[2 1 0]' '[3 1 0]']
        ['[0 1 1]' '[1 1 1]' '[2 1 1]' '[3 1 1]']
        ['[0 2 0]' '[1 2 0]' '[2 2 0]' '[3 2 0]']
        ['[0 2 1]' '[1 2 1]' '[2 2 1]' '[3 2 1]']
        ['[0 3 0]' '[1 3 0]' '[2 3 0]' '[3 3 0]']
        ['[0 3 1]' '[1 3 1]' '[2 3 1]' '[3 3 1]']
        ['[0 4 0]' '[1 4 0]' '[2 4 0]' '[3 4 0]']
        ['[0 4 1]' '[1 4 1]' '[2 4 1]' '[3 4 1]']
        ['[0 5 0]' '[1 5 0]' '[2 5 0]' '[3 5 0]']
        ['[0 5 1]' '[1 5 1]' '[2 5 1]' '[3 5 1]']
        ... ... ... ... ... ...
    ]
    where the 1st and 2nd digits are the x and y coords; the third labels A(0) B(1) sublattices. 
    This can be done by by numpy.reshape() the real-space correlation matrix in mps_indx
    """
    # step 1: Fourier transform in space
    # CtkA: Correlation data for sublattice A 
    # Cij[:, ::2, :] slices every other site along the a1-axis, starting from index 0 (even indices), assuming that sublattice A sites are at even indices.
    CtkA, kxlist, kylist = C_FT(Cij[:, ::2, :])
    
    # CtkB: Correlation data for sublattice B 
    # Cij[:, 1::2, :] slices every other site along the a1-axis, starting from index 1 (odd indices), assuming sublattice B sites are at odd indices.
    CtkB, kxlist, kylist = C_FT(Cij[:, 1::2, :])
    kX, kY = np.meshgrid(kxlist, kylist)
    if sublattice == 'A': # for 2-site unit cell
        if sub_only:
            CtkB *= 0
        else:
            CtkB *= np.exp(-1j*(kX+kY).T/3)[np.newaxis, :, :]
    elif sublattice == 'B':
        if sub_only:
            CtkA *= 0
        else:
            CtkA *= np.exp(1j*(kX+kY).T/3)[np.newaxis, :, :]
    else:
        raise ValueError("wrong sublattice")
    Ctk = CtkA + CtkB

    Lkx = len(kxlist)
    Lky = len(kylist)


    # step 2: linear prediction
    if n_lp > 0:
        times = np.arange(int(Cij.shape[0]*(1+n_lp)))*delta_t
        Ctk_lp = np.zeros((len(times), Ctk.shape[1], Ctk.shape[2]), dtype='complex')
        for kx in range(Lkx):
            for ky in range(Lky):
                Ctk_lp[:, kx, ky] = linear_prediction(Ctk[:, kx, ky], n_lp*Cij.shape[0])
    else:
        times = np.arange(int(Cij.shape[0]))*delta_t
        Ctk_lp = Ctk

    # step 3: Gaussian envelope
    # Multiplication in time corresponds to convolution in frequency
    Ctk_gauss = Ctk_lp * np.exp(-0.5*(times[:, np.newaxis, np.newaxis]/gauss_sigma)**2)

    # step 4: Fourier transform in time
    Akw, wlist = tempFT(Ctk_gauss, wmin, wmax, delta_t)

    return Akw, wlist, kxlist, kylist

    

# ----------------------------------------------------------
# ----------------- Debug make_spectrum() ----------------
# ----------------------------------------------------------
if __name__ == "__main__":
    from tenpy.tools import hdf5_io
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    # Step 1: Load the data
    hdf5_filename = './time_states/16TIME_j0opSzchi350dt0.010ExpMPO_L34defaultchi350_Jx-1.00Jy-1.00Jz-1.00JH0.00Jnnn0.00G0.00GP0.00Fx0.00Fy0.00Fz0.00Wp0.00.h5'

    with h5py.File(hdf5_filename, 'r') as f:
        measurements = hdf5_io.load_from_hdf5(f['measurements'])
        model_params = hdf5_io.load_from_hdf5(f['model_params'])
        j_unit_cell = int(hdf5_io.load_from_hdf5(f['j_unit_cell']))
        positions = hdf5_io.load_from_hdf5(f['pos'])
        # print(j_unit_cell + 2*model_params['Ly']*model_params['Lx']/2)
        time_keys = list(measurements.keys())

    Cij = np.zeros(((len(time_keys), model_params['Ly']*2, model_params['Lx'])), dtype=complex)
    # print(Cij)
    for t_indx, time in enumerate(time_keys):
        cij = np.array((measurements[time]['Cij_Sz']))
        cij = cij.reshape(model_params['Lx'],-1).T
        Cij[t_indx, :, :] = cij
    # Cij = measurements[0.19]['Cij_Sz']
    # print(Cij[0, :, :])

    
    Akw, wlist, kxlist, kylist = make_spectrum(Cij, delta_t=0.01, gauss_sigma=10, n_lp=0, wmin=0, wmax=100000, sublattice='A', sub_only=False)
    print(Akw.shape)

    # omega_index = 0
    # Akw_slice = Akw.real[omega_index, :, :]  # Shape: (kx_dim, ky_dim)


    fig, ax = plt.subplots(1,1, figsize=(8,8))
    # fig.set_size_inches(8,8)

    colmap = matplotlib.colormaps['magma']
    ax.imshow(np.roll(Akw.real[:, :, 0], 0, axis=1), cmap=colmap, origin='lower')
    # plt.pcolormesh(kx_grid, ky_grid, Akw_slice.real, shading='auto', cmap='viridis')
    # draw Akw for a fix omega in a contour!
    
    # fig.tight_layout(pad=0.2)
    plt.savefig("figSpec.pdf", dpi=300, bbox_inches='tight')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # N_sites = 2 * model_params['Ly'] * model_params['Lx']
    # dist_Map = np.zeros((N_sites, 2))

    # for i in range(N_sites):
    #     yR, xR = divmod(i, 2 * model_params['Ly'])

    #     if i % 2 == 0:
    #         dist_Map[i, 1] = xR / 2
    #         dist_Map[i, 0] = yR
    #     else:
    #         dist_Map[i, 1] = (1 / 3) + int(xR / 2)
    #         dist_Map[i, 0] = (1 / 3) + yR

    # dist_Map_with_Offset = dist_Map.copy()
    # dist_Map_with_Offset_norm_coord = dist_Map.copy()

    # ref_Site = int(N_sites / 2 + j_unit_cell)  # reference site in MPS (1-based)
    # rc1, rc2 = dist_Map[ref_Site - 1, :]
    # dist_Map_with_Offset[:, 0] -= rc1
    # dist_Map_with_Offset[:, 1] -= rc2

    # # to Euclidean basis
    # # a = (1/2, sqrt{3}/2); b = (1, 0)
    # # ria*a + rib*b = rix * x + riy * y
    # for i in range(N_sites):
    #     rix = 0.5 * dist_Map_with_Offset[i, 0] + dist_Map_with_Offset[i, 1]
    #     riy = 0.5 * math.sqrt(3) * dist_Map_with_Offset[i, 0]
    #     dist_Map_with_Offset_norm_coord[i, 0] = rix
    #     dist_Map_with_Offset_norm_coord[i, 1] = riy
    # # print(dist_Map_with_Offset_norm_coord)

    # Lys = dist_Map_with_Offset_norm_coord[:, 1].reshape(model_params['Lx'],-1).T
    # Lxs = dist_Map_with_Offset_norm_coord[:, 0].reshape(model_params['Lx'],-1).T
    # print(Lys)

    # Aijw, wlist = tempFT(Cij, wmin=0, wmax=10000, dt_corr=0.01)
    # # print(Aijw.shape)








    # # Define the desired k-points covering up to the second BZ
    # num_kx = 100
    # num_ky = 100
    # k1_vals = np.linspace(-2,2, num_kx)
    # k2_vals = np.linspace(-2,2, num_ky)

    # k1_grid, k2_grid = np.meshgrid(k1_vals, k2_vals, indexing='ij')

    # # Convert to Cartesian kx and ky using reciprocal lattice vectors
    # b1 = (2 * np.pi) * np.array([1, -1 / np.sqrt(3)])
    # b2 = np.array([0, (4 * np.pi) / np.sqrt(3)])

    # kx_grid = k1_grid * b1[0] + k2_grid * b2[0]
    # ky_grid = k1_grid * b1[1] + k2_grid * b2[1]

    # # Compute A(k, omega) at the desired omega
    # Akw_slice = np.zeros((num_kx, num_ky), dtype=complex)

    # omega = 0
    # for i in range(len(dist_Map_with_Offset_norm_coord)):
    #     r_i = dist_Map_with_Offset_norm_coord[i, :]  # Real-space position of site i
    #     C_i = Aijw[omega, :, :].reshape(N_sites, -1)[i]      # Correlation function at site i and desired omega
    #     phase = np.exp(-1j * (kx_grid * r_i[0] + ky_grid * r_i[1]))
    #     Akw_slice += C_i * phase


   





      