import numpy as np
import torch as tc
import pickle
import os

from torch.utils.checkpoint import checkpoint
from opt_einsum import contract as einsum


# tc.set_default_device('cuda')

ANGLE = 1.08 * np.pi / 180

g_1 = 4 * np.pi * np.sin(ANGLE / 2) * np.array([1 / np.sqrt(3), 1])
g_2 = 8 * np.pi * np.sin(ANGLE / 2) * np.array([-1 / np.sqrt(3), 0])

# Generate right triangle momentum grid

b_1 = 4 * np.pi * np.sin(ANGLE / 2) * np.array([1 / np.sqrt(3), 0])
b_2 = 4 * np.pi * np.sin(ANGLE / 2) * np.array([0, 1 / 3])

K_POINTS = []
K_X = 2
K_Y = 4

for n_x in range(K_X):
    for n_y in range(K_Y):
        u_x = n_x / K_X
        u_y = n_y / K_Y

        if u_x < u_y:
            k = b_1 * u_x + b_2 * u_y

            K_POINTS.append(-k)
            K_POINTS.append(k)

K_POINTS = np.array(K_POINTS)


CELLS = np.array([[0.0, 0.0], g_1, -g_1, g_2, -g_2, g_1 + g_2, -g_1 - g_2])
HARMONICS = np.array([[0.0, 0.0], -g_1, -g_1 - g_2])

a_1 = 2.846e-2
a_2 = 3.482e-2

TOTAL_SIZE = CELLS.shape[0] * \
    (HARMONICS.shape[0] + 1) * K_POINTS.shape[0] * 2 * 2

FILLING = 2.0 * K_POINTS.shape[0] * 2

TOLERANCE = 1e-5
STEP = 1e-3

N_GAPS = 100

DOUBLES_SIZE = 16

# Potential in momentum space


def momentumPotential(k_x, k_y, band_p, band_q, spin_p, spin_q):
    result = np.zeros((1, TOTAL_SIZE, TOTAL_SIZE), dtype=np.cfloat)

    q = np.sqrt(k_x ** 2 + k_y ** 2)

    result += np.logical_not(q == 0.0).astype(int) * 0.27 * 2 * np.pi * np.nan_to_num(
        np.tanh(162 * q) / q) * (8 * np.sin(ANGLE / 2) ** 2) / np.sqrt(3)

    result += np.logical_and(np.logical_and(band_p == band_q,
                             spin_p == spin_q), q == 0.0).astype(int) * 0.0062

    return result

# Calculate kinetic tensor


def getKineticTensor() -> np.ndarray:
    K = np.zeros((TOTAL_SIZE, TOTAL_SIZE), dtype=np.cfloat)

    for index in range(0, TOTAL_SIZE, 4):
        k_x, k_y = getMomentum(np.array(index))

        q = np.sqrt(k_x ** 2 + k_y ** 2)

        K[index, index] = - 0.763 * q
        K[index + 1, index + 1] = - 0.763 * q
        K[index + 2, index + 2] = 0.763 * q
        K[index + 3, index + 3] = 0.763 * q

    U = np.zeros((CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2, TOTAL_SIZE -
                 CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2), dtype=np.cfloat)

    for harmonic in range(HARMONICS.shape[0]):
        for index_1 in range(0, CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2, 4):
            index_2 = index_1 + harmonic * \
                CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2

            kx_1, ky_1 = getMomentum(np.array(index_1))
            kx_2, ky_2 = getMomentum(
                np.array(index_2 + CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2))

            # Calculate potential matrix in the new basis

            H_1 = np.zeros((4, 4), dtype=np.cfloat)
            H_2 = np.zeros((4, 4), dtype=np.cfloat)
            V = np.zeros((4, 4), dtype=np.cfloat)

            A_1 = a_1
            A_2 = a_2

            if harmonic == 1.0:
                A_2 *= np.exp(- 2.0j * np.pi / 3)
            elif harmonic == 2.0:
                A_2 *= np.exp(2.0j * np.pi / 3)

            V[0:2, 0:2] = A_1
            V[2:4, 2:4] = A_1
            V[0:2, 2:4] = A_2
            V[2:4, 0:2] = A_2

            C_1 = np.sqrt(kx_1 ** 2 + ky_1 ** 2) / (kx_1 + 1.0j * ky_1)
            C_2 = np.sqrt(kx_2 ** 2 + ky_2 ** 2) / (kx_2 + 1.0j * ky_2)

            H_1[0, 0] = -C_1
            H_1[0, 2] = C_1
            H_1[1, 1] = -C_1
            H_1[1, 3] = C_1

            H_2[0, 0] = -C_2
            H_2[0, 2] = C_2
            H_2[1, 1] = -C_2
            H_2[1, 3] = C_2

            H_1[2, 1] = 1
            H_1[2, 3] = 1
            H_1[3, 0] = 1
            H_1[3, 2] = 1

            H_2[2, 1] = 1
            H_2[2, 3] = 1
            H_2[3, 0] = 1
            H_2[3, 2] = 1

            U[index_1:index_1 + 4, index_2:index_2 +
                4] = np.linalg.inv(H_1) @ V @ H_2

    K[0:CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2,
        CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2:] = U
    K[CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2:, 0:CELLS.shape[0]
        * K_POINTS.shape[0] * 2 * 2:] = np.conjugate(np.transpose(U))

    K /= K_POINTS.shape[0]

    return tc.tensor(K, dtype=tc.cfloat, requires_grad=False)


# Calculate potential tensor


def getPotentialTensor() -> np.ndarray:

    V = np.zeros((TOTAL_SIZE, TOTAL_SIZE, TOTAL_SIZE), dtype=np.cfloat)

    # Seperate the potential in chunks and process seperately

    for chunk in range(TOTAL_SIZE):

        print(f"Calculating potential chunk {chunk}...")

        V_chunk = np.ones((1, TOTAL_SIZE, TOTAL_SIZE), dtype=np.cfloat)
        indices = np.indices((1, TOTAL_SIZE, TOTAL_SIZE))
        indices[0] += chunk

        band_p = np.floor(np.remainder(indices[0], 4) / 2)
        band_q = np.floor(np.remainder(indices[1], 4) / 2)

        spin_p = np.remainder(indices[0], 2)
        spin_q = np.remainder(indices[0], 2)

        kx_p, ky_p = getMomentum(indices[0])
        kx_q, ky_q = getMomentum(indices[1])
        kx_G, ky_G = getMomentum(indices[2])

        layer_p = getLayer(indices[0])
        layer_q = getLayer(indices[1])
        layer_G = getLayer(indices[2])

        ky_p += 4 * np.pi * np.sin(ANGLE * (2 * layer_p - 1) / 2) / 3
        ky_q += 4 * np.pi * np.sin(ANGLE * (2 * layer_q - 1) / 2) / 3
        ky_G += 4 * np.pi * np.sin(ANGLE * (2 * layer_G - 1) / 2) / 3

        V_chunk *= kx_p - kx_q == kx_G
        V_chunk *= ky_p - ky_q == ky_G

        V_chunk *= np.sqrt(momentumPotential(kx_G, ky_G, band_p, band_q,
                                             spin_p, spin_q) / (2 * K_POINTS.shape[0] ** 6))

        V[chunk, :, :] = V_chunk

    return tc.tensor(V, dtype=tc.cfloat, requires_grad=False)


# Map number operator signs


def getNumberSigns():
    L = np.zeros((TOTAL_SIZE, ), dtype=np.cfloat)
    indices = np.indices((TOTAL_SIZE, ))

    band_p = np.floor(np.remainder(indices[0], 4) / 2)

    L -= band_p == 0.0
    L += band_p == 1.0

    return tc.tensor(L, dtype=tc.cfloat, requires_grad=False)


# ECCSD energy equation

def eccsd_energy(K, V, T1, X1, T2, X2):
    V_conj = tc.conj_physical(V)

    result = 0.0

    result += checkpoint(einsum, 'ik,jk,ij', K, T1, X1)
    result += checkpoint(einsum, 'ija,kla,ij,kl', V_conj, V, X1, T1) / 2.0
    result -= checkpoint(einsum, 'ika,kla,ij,jl', V_conj, V, X1, T1) / 2.0

    result -= checkpoint(einsum, 'ija,mna,km,ln,ik,jl',
                         V_conj, V, T1, T1, X1, X1)
    result += checkpoint(einsum, 'ija,mna,km,ln,ij,kl',
                         V_conj, V, T1, T1, X1, X1) / 2.0

    result += checkpoint(einsum, 'km,ln,ija,mna,ijb,klb',
                         T1, T1, V_conj, V, X2, X2) / 2.0

    result -= checkpoint(einsum, 'ima,mna,jkb,lnb,ikc,jlc',
                         V_conj, V, T2, T2, X2, X2) / 12.0
    result -= checkpoint(einsum, 'ima,mna,jkb,lnb,ilc,jkc',
                         V_conj, V, T2, T2, X2, X2) / 12.0

    result += checkpoint(einsum, 'mna,ija,klc,mnc,ik,jl',
                         V_conj, V, T2, T2, X1, X1) / 2.0
    result -= checkpoint(einsum, 'mna,ija,klc,mnc,ijc,klc',
                         V_conj, V, T2, T2, X2, X2) / 4.0

    result -= checkpoint(einsum, 'im,ija,kla,jkb,lmb', K, X2, X2, T2, T2) / 3.0
    result += checkpoint(einsum, 'im,ija,kla,jmb,klb',
                         K, X2, X2, T2, T2) / 6.0

    result += 2.0 * \
        checkpoint(einsum, 'opa,ija,lmb,npb,ko,ik,jlc,nmc',
                   V_conj, V, T2, T2, T1, X1, X2, X2) / 3.0
    result -= checkpoint(einsum, 'opa,ija,lmb,npb,ko,ik,jnc,lmc',
                         V_conj, V, T2, T2, T1, X1, X2, X2) / 3.0
    result -= 2.0 * \
        checkpoint(einsum, 'opa,ija,lmb,npb,ko,im,jnc,klc',
                   V_conj, V, T2, T2, T1, X1, X2, X2)
    result -= 2.0 * \
        checkpoint(einsum, 'opa,ija,lmb,npb,ko,im,jlc,knc',
                   V_conj, V, T2, T2, T1, X1, X2, X2)
    result -= checkpoint(einsum, 'opa,ija,lmb,npb,ko,im,jkc,lnc',
                         V_conj, V, T2, T2, T1, X1, X2, X2)
    result -= checkpoint(einsum, 'opa,ija,lmb,npb,ko,kn,ijc,lmc',
                         V_conj, V, T2, T2, T1, X1, X2, X2) / 2.0
    result -= checkpoint(einsum, 'opa,ija,lmb,npb,ko,ij,klc,mnc',
                         V_conj, V, T2, T2, T1, X1, X2, X2) / 3.0
    result += checkpoint(einsum, 'opa,ija,lmb,npb,ko,ij,knc,lmc',
                         V_conj, V, T2, T2, T1, X1, X2, X2) / 6.0
    result -= checkpoint(einsum, 'opa,ija,lmb,npb,ko,il,jm,kn',
                         V_conj, V, T2, T2, T1, X1, X1, X1)

    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ijd,kld,mne,ope',
                         V_conj, V, T2, T2, T2, T2, X2, X2, X2, X2) / 6.0
    result -= checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ijd,kld,mpe,noe',
                         V_conj, V, T2, T2, T2, T2, X2, X2, X2, X2) / 12.0
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ijd,knd,lme,ope',
                         V_conj, V, T2, T2, T2, T2, X2, X2, X2, X2) / 2.0
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ijd,knd,loe,mpe',
                         V_conj, V, T2, T2, T2, T2, X2, X2, X2, X2) / 2.0
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ijd,mpd,lke,one',
                         V_conj, V, T2, T2, T2, T2, X2, X2, X2, X2) / 8.0
    result -= checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,imd,jkd,lne,ope',
                         V_conj, V, T2, T2, T2, T2, X2, X2, X2, X2)
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ipd,jnd,kle,moe',
                         V_conj, V, T2, T2, T2, T2, X2, X2, X2, X2)
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,imd,jld,kpe,one',
                         V_conj, V, T2, T2, T2, T2, X2, X2, X2, X2) / 4.0
    result -= checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ild,jpd,kme,noe',
                         V_conj, V, T2, T2, T2, T2, X2, X2, X2, X2) / 9.0
    result -= checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ikd,jnd,lme,ope',
                         V_conj, V, T2, T2, T2, T2, X2, X2, X2, X2) / 9.0
    result -= checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,imd,jpd,kle,noe',
                         V_conj, V, T2, T2, T2, T2, X2, X2, X2, X2) / 36.0
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ij,mn,lkd,opd',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2) / 2.0
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ij,kp,lmd,ond',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2) / 2.0
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ij,mp,lkd,ond',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2) / 8.0
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ik,jl,mod,npd',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2) / 3.0
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ik,jl,mpd,nod',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2) / 6.0
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ik,jp,mld,nod',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2) / 2.0
    result -= checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,im,jp,lkd,nod',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2) / 4.0
    result += 2.0 * checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ik,ln,jmd,opd',
                               V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2)
    result += 2.0 * checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ik,mn,jld,pod',
                               V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2)
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,il,kp,jmd,ond',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2)
    result -= checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ik,lp,jmd,ond',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2)
    result -= checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ik,mp,jld,ond',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X2, X2)
    result -= checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ij,kn,lp,mo',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X1, X1) / 6.0
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ij,mp,ln,ko',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X1, X1) / 12.0
    result += 2.0 * checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ik,jp,mo,ln',
                               V_conj, V, T2, T2, T2, T2, X1, X1, X1, X1)
    result -= checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ik,jn,lp,mo',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X1, X1)
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,ik,jn,lo,mp',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X1, X1)
    result += checkpoint(einsum, 'ija,rsa,nob,psb,klc,mrc,im,jp,lo,kn',
                         V_conj, V, T2, T2, T2, T2, X1, X1, X1, X1) / 2.0

    return result


# ECCSD particle number equation


def eccsd_particle_number(L, T1, X1, T2, X2):
    result = 0.0

    result -= checkpoint(einsum, 'i,ij,ij', L, T1, X1)

    result += checkpoint(einsum, 'i,ijp,klp,ijq,klq', L, T2, T2, X2, X2) / 2.0
    result += checkpoint(einsum, 'i,ikp,jlp,ilq,jkq', L, T2, T2, X2, X2) / 3.0

    return result

# ECCSD excitation energy equation


def eccsd_excitation_energy(K, V, T1, X1, T2, X2, x, y):
    V_conj = tc.conj_physical(V)

    result = 0.0

    result += K[x, y]

    if x == y:
        result += einsum('ia,ia', V_conj[x, :, :], V[y, :, :])

    result -= 2.0 * einsum('ia,ma,km,ik', V_conj[x, :, :], V[y, :, :], T1, X1)
    result -= einsum('ija,ma,m,ij', V_conj, V[y, :, :], T1[x, :], X1)

    result += 2.0 * einsum('ija,oa,kb,lob,il,jk', V_conj,
                           V[y, :, :], T2[x, :, :], T2, X1, X1)
    result += einsum('ija,oa,kb,lob,ijc,klc', V_conj,
                     V[y, :, :], T2[x, :, :], T2, X2, X2)

    result += einsum('ija,oa,ob,klb,il,jk', V_conj,
                     V[y, :, :], T2[:, x, :], T2, X1, X1)
    result += einsum('ija,oa,ob,klb,ijc,klc', V_conj,
                     V[y, :, :], T2[:, x, :], T2, X2, X2) / 2.0

    result += 2.0 * einsum('ia,oa,klb,mob,ikc,lmc',
                           V_conj[x, :, :], V[y, :, :], T2, T2, X2, X2) / 3.0
    result -= einsum('ia,oa,klb,mob,imc,klc',
                     V_conj[x, :, :], V[y, :, :], T2, T2, X2, X2) / 3.0

    return result


# ECCSD superconducting gap

def eccsd_superconducting_gap(X1, T2, X2, x, y):

    result = 0.0

    result += einsum('ia,jka,lb,nmb,in,jl,km',
                     T2[x, :, :], T2, T2[y, :, :], T2, X1, X1, X1) / 4.0
    result -= einsum('ia,jka,lb,nmb,kn,jm,il',
                     T2[x, :, :], T2, T2[y, :, :], T2, X1, X1, X1) / 8.0
    result -= einsum('ia,jka,lb,nmb,il,jkc,mnc',
                     T2[x, :, :], T2, T2[y, :, :], T2, X1, X2, X2) / 8.0

    result += einsum('ia,jka,lb,nmb,im,jlc,knc',
                     T2[x, :, :], T2, T2[y, :, :], T2, X1, X2, X2) / 2.0
    result += einsum('ia,jka,lb,nmb,il,jmc,knc',
                     T2[x, :, :], T2, T2[y, :, :], T2, X1, X2, X2) / 2.0

    return result


# Match index to mapped properties

def getProperties(index):
    layer = (index >= CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2).astype(int)

    harmonic = np.floor((index - layer * CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2) / (
        CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2)).astype(int)

    cell = np.floor((index - layer * CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2 - harmonic *
                    CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2) / (K_POINTS.shape[0] * 2 * 2)).astype(int)

    k_point = np.floor((index - layer * CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2 - harmonic *
                       CELLS.shape[0] * K_POINTS.shape[0] * 2 * 2 - cell * K_POINTS.shape[0] * 2 * 2) / 4).astype(int)

    band = np.floor(np.remainder(index, 4) / 2)
    spin = np.remainder(index, 2)

    return layer, harmonic, cell, k_point, band, spin


# Match index to layer

def getLayer(index):
    layer, harmonic, cell, k_point, band, spin = getProperties(index)

    return layer


# Match index to momentum

def getMomentum(index):
    layer, harmonic, cell, k_point, band, spin = getProperties(index)

    k_x = K_POINTS[k_point, 0] + CELLS[cell, 0] + HARMONICS[harmonic, 0]
    k_y = K_POINTS[k_point, 1] + CELLS[cell, 1] + HARMONICS[harmonic, 1]

    return k_x, k_y

# Symetrize rank 2 tensor and keep the real part


def processSingles(tensor):
    result = 0.0

    result += einsum('ij->ij', tensor)
    result -= einsum('ij->ji', tensor)

    return result / 2.0


# Symetrize rank 3 tensor and keep the real part


def processDoubles(tensor):
    result = 0.0

    result += einsum('ijk->ijk', tensor)
    result -= einsum('ijk->jik', tensor)

    return result / 2.0


# Create initial amplitude tensors

def getInitiators():
    I1 = np.ones((TOTAL_SIZE, TOTAL_SIZE), dtype=np.cfloat)
    indices = np.indices((TOTAL_SIZE, TOTAL_SIZE))

    I1 *= indices[0] < indices[1]
    I1 *= 1e-2 * 2.0

    I2 = np.ones((TOTAL_SIZE, TOTAL_SIZE, DOUBLES_SIZE), dtype=np.cfloat)
    indices = np.indices((TOTAL_SIZE, TOTAL_SIZE, DOUBLES_SIZE))

    I2 *= indices[0] < indices[1]
    I2 *= 1e-4 * 2.0

    return I1, I2

# Anti-symmetrize and compress potential tensor


def processPotential(V_raw):

    # Do singular value decomposition
    V_uncompressed = (einsum('ijk->ijk', V_raw) - einsum('ijk->jik', V_raw))/2

    G = einsum('abi,abj->ij', tc.conj_physical(V_uncompressed), V_uncompressed)
    eigenvalues, eigenvectors = tc.linalg.eig(G)

    # Save singular values
    np.savetxt("Compression Singular Values.csv",
               tc.abs(eigenvalues).cpu().numpy(), delimiter=', ')

    # Select largest singular values and eigenvectors
    largest_indices = tc.topk(tc.abs(eigenvalues), DOUBLES_SIZE)[1].tolist()

    U = eigenvectors[:, largest_indices]

    # Build compressed tensor
    V_compressed = tc.einsum(
        'ab,ija->ijb', tc.conj_physical(U), V_uncompressed)

    # Retain anti-symmetry
    return (einsum('ijk->ijk', V_compressed) - einsum('ijk->jik', V_compressed))/2


if __name__ == "__main__":

    print("Building tensors...")

    # Build or load the potential, kinetic tensors and number operator sign vector

    if os.path.exists('V.bin'):
        V_raw = pickle.load(open('V.bin', 'rb'))
    else:
        V_raw = getPotentialTensor()

        pickle.dump(V_raw, open('V.bin', 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    V = processPotential(V_raw)
    K = getKineticTensor()
    L = getNumberSigns()

    del V_raw

    # Allocate the amplitudes

    print("Allocating or loading amplitudes...")

    if os.path.exists('T1.bin') and os.path.exists('X1.bin') and os.path.exists('T2.bin') and os.path.exists('X2.bin'):
        T1 = pickle.load(open('T1.bin', 'rb'))
        X1 = pickle.load(open('X1.bin', 'rb'))
        T2 = pickle.load(open('T2.bin', 'rb'))
        X2 = pickle.load(open('X2.bin', 'rb'))
    else:
        I1, I2 = getInitiators()

        T1 = tc.tensor(I1, dtype=tc.cfloat, requires_grad=True)
        X1 = tc.tensor(I1, dtype=tc.cfloat, requires_grad=True)
        T2 = tc.tensor(I2, dtype=tc.cfloat, requires_grad=True)
        X2 = tc.tensor(I2, dtype=tc.cfloat, requires_grad=True)

        del I1, I2

    parameters = [T1, X1, T2, X2]

    # Define the minimization function

    def minimization_function(parameters):

        T1, X1, T2, X2 = parameters

        T2_skew = processDoubles(T2)
        X2_skew = processDoubles(X2)
        T1_skew = processSingles(T1)
        X1_skew = processSingles(X1)

        particle_number = eccsd_particle_number(
            L, T1_skew, X1_skew, T2_skew, X2_skew)
        energy = eccsd_energy(K, V, T1_skew, X1_skew, T2_skew, X2_skew)

        del T2_skew, X2_skew, T1_skew, X1_skew

        return energy.item(), particle_number.item(), tc.real(energy + (particle_number - FILLING) ** 2)

    # Run optimization

    print("Running simulation...")

    # tc.cuda.empty_cache()

    if os.path.exists('convergence.bin'):
        convergence_data, previous_energy, loop = pickle.load(
            open('convergence.bin', 'rb'))
    else:
        previous_energy = np.inf
        convergence_data = []
        loop = 0

    if not os.path.exists('Convergence Data.csv'):

        optimizer = tc.optim.SGD(parameters, lr=STEP)

        while True:

            energy, particle_number, variable = minimization_function(
                parameters)

            optimizer.zero_grad()
            variable.backward()
            optimizer.step()

            # tc.cuda.empty_cache()

            loop += 1

            print(f"Progress is {loop}!")

            # Check for convergence
            energy_change = np.abs(1 - energy / previous_energy)
            print(f"Current energy change is {energy_change}!")

            convergence_data.append(
                [np.real(energy), np.real(particle_number)])

            if energy_change < TOLERANCE:
                break

            previous_energy = energy

            pickle.dump(T1, open('T1.bin', 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(X1, open('X1.bin', 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(T2, open('T2.bin', 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(X2, open('X2.bin', 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)

            pickle.dump([convergence_data, previous_energy, loop], open(
                'convergence.bin', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        # Save data
        np.savetxt("Convergence Data.csv", np.array(
            convergence_data), delimiter=', ')

    with tc.no_grad():
        T2_skew = processDoubles(T2)
        X2_skew = processDoubles(X2)
        T1_skew = processSingles(T1)
        X1_skew = processSingles(X1)

        print("Calculating superconducting gaps...")

        if not os.path.exists("Gap Data.csv"):

            # Calculate superconducting order parameters
            order_values = []

            for a in range(0, TOTAL_SIZE):
                layer_a, harmonic_a, cell_a, k_point_a, band_a, spin_a = getProperties(
                    np.array(a))
                kx_a, ky_a = getMomentum(np.array(a))

                ky_a += 4 * np.pi * np.sin(ANGLE * (2 * layer_a - 1) / 2) / 3

                for b in range(0, TOTAL_SIZE):
                    layer_b, harmonic_b, cell_b, k_point_b, band_b, spin_b = getProperties(
                        np.array(b))
                    kx_b, ky_b = getMomentum(np.array(b))

                    ky_b += 4 * np.pi * \
                        np.sin(ANGLE * (2 * layer_b - 1) / 2) / 3

                    if spin_a == 1.0 and spin_b == 0.0 and kx_b == - kx_a and ky_b == - ky_a:

                        value = eccsd_superconducting_gap(
                            X1_skew, T2_skew, X2_skew, a, b)

                        # Add 12-fold symmetric values
                        for angle in range(0, 360, 30):
                            angle *= np.pi / 180

                            order_values.append(
                                [kx_a*np.cos(angle) - ky_a*np.sin(angle), kx_a*np.sin(angle) + ky_a*np.cos(angle), value.item()])

                        # tc.cuda.empty_cache()

                        print(f"Progress is {a}, {b}!")

            # Calculate superconducting gaps
            gap_data = []

            for n_x in range(N_GAPS + 1):
                for n_y in range(N_GAPS + 1):

                    u_x = n_x / N_GAPS
                    u_y = n_y / N_GAPS

                    if u_x >= u_y:
                        k = b_1 * u_x + b_2 * u_y

                        gap = 0.0

                        for i in range(len(order_values)):

                            kx_i, ky_i, order_parameter = order_values[i]

                            q = np.sqrt((kx_i - k[0]) **
                                        2 + (ky_i - k[1]) ** 2)

                            if q == 0.0:
                                gap -= 0.0062 * order_parameter
                            else:
                                gap -= 0.27 * 2 * np.pi * np.nan_to_num(np.tanh(162 * q) / q) * (
                                    8 * np.sin(ANGLE / 2) ** 2) / np.sqrt(3) * order_parameter

                        gap_data.append(
                            [np.real(k[0]), np.real(k[1]), np.real(gap)])

            # Save data
            np.savetxt("Superconducting Gaps.csv", np.array(gap_data), delimiter=', ')

        print("Calculating excited states...")

        # Calculate excited energies
        size = 4 * (HARMONICS.shape[0] + 1) * CELLS.shape[0]

        if os.path.exists('excitations.bin'):
            band_data, matrix, point, index_a, a = pickle.load(
                open('excitations.bin', 'rb'))
        else:
            matrix = np.zeros((size, size), dtype=np.cfloat)
            band_data = []
            index_a = -1
            point = 1
            a = 0

        while point < K_POINTS.shape[0]:

            while a < TOTAL_SIZE:
                layer_a, harmonic_a, cell_a, k_point_a, band_a, spin_a = getProperties(
                    np.array(a))

                if not point == k_point_a:
                    continue

                index_a += 1
                index_b = -1

                for b in range(TOTAL_SIZE):
                    layer_b, harmonic_b, cell_b, k_point_b, band_b, spin_b = getProperties(
                        np.array(b))

                    if not point == k_point_b:
                        continue

                    index_b += 1

                    matrix[index_a, index_b] = eccsd_excitation_energy(
                        K, V, T1_skew, X1_skew, T2_skew, X2_skew, a, b).item()

                    # tc.cuda.empty_cache()

                    print(f"Progress is {point}, {index_a}, {index_b}!")

                a += 1

                pickle.dump([band_data, matrix, point, index_a, a], open(
                    'excitations.bin', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            energies = np.linalg.eigvals(matrix)
            data = sorted(energies, key=abs)

            band_data.append([K_POINTS[point][0], K_POINTS[point][1], np.real(data[0]), np.real(
                data[1]), np.real(data[2]), np.real(data[3]), np.real(data[4]), np.real(data[5])])

            matrix = np.zeros((size, size), dtype=np.cfloat)
            index_a = -1
            a = 0

            point += 2

        # Save data
        np.savetxt("Band Data.csv", np.array(band_data), delimiter=', ')
