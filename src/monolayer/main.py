import numpy as np
import torch as tc
import pickle
import os

from torch.utils.checkpoint import checkpoint
from opt_einsum import contract as einsum

g_1 = 2 * np.pi * np.array([1, np.sqrt(3)]) / 3
g_2 = 2 * np.pi * np.array([1, -np.sqrt(3)]) / 3

d_1 = np.array([1, -np.sqrt(3)]) / 2
d_2 = np.array([1, np.sqrt(3)]) / 2
d_3 = np.array([-1, 0])

# Generate right triangle momentum grid

K_POINTS = []
K_SIZE = 8

for n_x in range(K_SIZE):
    for n_y in range(K_SIZE):
        u_x = n_x / K_SIZE
        u_y = n_y / K_SIZE

        b_1 = (g_1 - g_2) / 2
        b_2 = (g_1 + g_2) / 2

        k = b_1 * u_x + b_2 * u_y

        if u_x < u_y:
            K_POINTS.append(k)

K_POINTS = np.array(K_POINTS)

TOTAL_SIZE = 2 * K_POINTS.shape[0]

FILLING = 1.0 * K_POINTS.shape[0]
TOLERANCE = 1e-5
STEP = 5e-3

# Band energy


def band(kx, ky) -> float:
    return np.abs(np.exp(1.0j * (kx * d_1[0] + ky * d_1[1])) + np.exp(1.0j * (kx * d_2[0] + ky * d_2[1])) + np.exp(1.0j * (kx * d_3[0] + ky * d_3[1])))


# Potential in momentum space


def momentumPotential(k_x, band_p, band_q):
    result = np.zeros((1, TOTAL_SIZE, TOTAL_SIZE, TOTAL_SIZE), dtype=float)

    result += (band_p == band_q).astype(int) * \
        (2.0 * 1.34 * np.cos(k_x) + 2.0 * 3.3 + 2.0 * 4.22)
    result -= np.logical_not(band_p == band_q).astype(int) * \
        (2.0 * 1.34 * np.cos(k_x))

    return result


# Calculate kinetic tensor


def getKineticTensor() -> np.ndarray:
    K = np.ones((TOTAL_SIZE, TOTAL_SIZE), dtype=float)
    indices = np.indices((TOTAL_SIZE, TOTAL_SIZE))

    band_p, kx_p, ky_p = getProperties(indices[0])
    band_q, kx_q, ky_q = getProperties(indices[1])

    K *= band_p == band_q
    K *= kx_p == kx_q
    K *= ky_p == ky_q

    K *= (band_p == 1.0).astype(int) - (band_p == 0.0).astype(int)
    K *= band(kx_p, ky_p) / K_POINTS.shape[0]

    return tc.tensor(K, dtype=tc.float, requires_grad=False)

# Calculate potential tensor


def getPotentialTensor() -> np.ndarray:

    V = np.zeros((TOTAL_SIZE, TOTAL_SIZE, TOTAL_SIZE, TOTAL_SIZE), dtype=float)

    # Seperate the potential in chunks and process seperately

    for chunk in range(TOTAL_SIZE):

        print(f"Calculating potential chunk {chunk}...")

        V_chunk = np.ones((1, TOTAL_SIZE, TOTAL_SIZE, TOTAL_SIZE), dtype=float)

        indices = np.indices((1, TOTAL_SIZE, TOTAL_SIZE, TOTAL_SIZE))

        indices[0] += chunk

        band_p, kx_p, ky_p = getProperties(indices[0])
        band_q, kx_q, ky_q = getProperties(indices[1])
        band_r, kx_r, ky_r = getProperties(indices[2])
        band_s, kx_s, ky_s = getProperties(indices[3])

        V_chunk *= kx_p + kx_q == kx_r + kx_s
        V_chunk *= ky_p + ky_q == ky_r + ky_s
        V_chunk *= band_p == band_s
        V_chunk *= band_q == band_r

        V_chunk *= (momentumPotential(kx_q - kx_r, band_p, band_r) -
                    momentumPotential(kx_p - kx_r, band_p, band_r)) / (2 * K_POINTS.shape[0] ** 3)

        V[chunk, :, :, :] = V_chunk

    return tc.tensor(V, dtype=tc.float, requires_grad=False)

# Map number operator signs


def getNumberSigns():
    L = np.zeros((TOTAL_SIZE, ), dtype=float)
    indices = np.indices((TOTAL_SIZE, ))

    band_p, kx_p, ky_p = getProperties(indices[0])

    L -= band_p == 0.0
    L += band_p == 1.0

    return tc.tensor(L, dtype=tc.float, requires_grad=False)


# ECCSD energy equation

def eccsd_energy(K, V, T1, X1, T2, X2):
    result = 0.0

    result += checkpoint(einsum, 'ik,jk,ij', K, T1, X1)
    result += checkpoint(einsum, 'ijkl,ij,kl', V, X1, T1) / 2.0
    result -= checkpoint(einsum, 'ikkl,ij,jl', V, X1, T1) / 2.0

    result -= checkpoint(einsum, 'ijmn,km,ln,ik,jl', V, T1, T1, X1, X1)
    result += checkpoint(einsum, 'ijmn,km,ln,ij,kl', V, T1, T1, X1, X1) / 2.0
    result += checkpoint(einsum, 'ijmn,km,ln,ijkl', V, T1, T1, X2) / 2.0
    result += checkpoint(einsum, 'immn,jkln,ijkl', V, T2, X2) / 12.0
    result += checkpoint(einsum, 'ijmn,klmn,ik,jl', V, T2, X1, X1) / 2.0
    result -= checkpoint(einsum, 'ijmn,klmn,ijkl', V, T2, X2) / 4.0

    result -= checkpoint(einsum, 'im,jklm,ijkl', K, T2, X2) / 6.0
    result += checkpoint(einsum, 'ijop,ko,lmnp,ik,jlmn',
                         V, T1, T2, X1, X2) / 3.0
    result += checkpoint(einsum, 'ijop,ko,lmnp,im,jkln', V, T1, T2, X1, X2)
    result += checkpoint(einsum, 'ijop,ko,lmnp,km,ijln',
                         V, T1, T2, X1, X2) / 2.0
    result -= checkpoint(einsum, 'ijop,ko,lmnp,ij,klmn',
                         V, T1, T2, X1, X2) / 6.0
    result += checkpoint(einsum, 'ijop,ko,lmnp,il,jm,kn',
                         V, T1, T2, X1, X1, X1)

    result += checkpoint(einsum, 'ijrs,klmr,nops,ijkl,mnop',
                         V, T2, T2, X2, X2) / 12.0
    result += checkpoint(einsum, 'ijrs,klmr,nops,ijkn,lmop',
                         V, T2, T2, X2, X2) / 8.0
    result -= checkpoint(einsum, 'ijrs,klmr,nops,imop,jkln',
                         V, T2, T2, X2, X2) / 4.0
    result -= checkpoint(einsum, 'ijrs,klmr,nops,iklm,jnop',
                         V, T2, T2, X2, X2) / 36.0
    result += checkpoint(einsum, 'ijrs,klmr,nops,ij,kn,lmop',
                         V, T2, T2, X1, X1, X2) / 8.0

    result += checkpoint(einsum, 'ijrs,klmr,nops,ik,jm,lnop',
                         V, T2, T2, X1, X1, X2) / 6.0
    result += checkpoint(einsum, 'ijrs,klmr,nops,ik,jo,lmnp',
                         V, T2, T2, X1, X1, X2) / 4.0
    result += checkpoint(einsum, 'ijrs,klmr,nops,ik,ln,jmop',
                         V, T2, T2, X1, X1, X2)
    result -= checkpoint(einsum, 'ijrs,klmr,nops,lp,mn,ijko',
                         V, T2, T2, X1, X1, X2) / 4.0
    result += checkpoint(einsum, 'ijrs,klmr,nops,ij,kn,lp,mo',
                         V, T2, T2, X1, X1, X1, X1) / 12.0

    result += checkpoint(einsum, 'ijrs,klmr,nops,ik,jn,lo,mp',
                         V, T2, T2, X1, X1, X1, X1) / 2.0

    return result


# ECCSD particle number equation


def eccsd_particle_number(L, T1, X1, T2, X2):
    result = 0.0

    result -= checkpoint(einsum, 'i,ij,ij', L, T1, X1)
    result += checkpoint(einsum, 'i,ijkl,ijkl', L, T2, X2) / 6.0

    return result

# ECCSD excitation energy equation


def eccsd_excitation_energy(K, V, T1, X1, T2, X2, a, b):
    result = 0.0

    result += K[a, b]

    if a == b:
        result += einsum('ii', V[a, :, b, :])

    result -= 2.0 * \
        einsum('im,km,ik', V[a, :, b, :], T1, X1)
    result -= einsum('ijm,m,ij', V[:, :, b, :], T1[a, :], X1)

    result += einsum('ijo,klo,il,jk', V[:, :, b, :], T2[a, :, :, :], X1, X1)
    result += einsum('ijo,klo,ijkl', V[:, :, b, :], T2[a, :, :, :], X2) / 2.0
    result += einsum('io,klmo,iklm', V[a, :, b, :], T2, X2) / 3.0

    result *= K_POINTS.shape[0]

    return result

# Match index to momentum and band values


def getProperties(index):
    band = np.floor(index / K_POINTS.shape[0])

    n_k = (index - band * K_POINTS.shape[0]).astype(int)

    return band, K_POINTS[n_k, 0], K_POINTS[n_k, 1]


# Symetrize rank 2 tensor


def antisymmetrizeRank2Tensor(tensor):
    result = 0.0

    result += einsum('ij->ij', tensor)
    result -= einsum('ij->ji', tensor)

    return result / 2.0


# Symetrize rank 4 tensor


def antisymmetrizeRank4Tensor(tensor):
    result = 0.0

    result += einsum('ijkl->ijkl', tensor)
    result -= einsum('ijkl->ijlk', tensor)
    result += einsum('ijkl->iljk', tensor)
    result -= einsum('ijkl->ilkj', tensor)
    result += einsum('ijkl->iklj', tensor)
    result -= einsum('ijkl->ikjl', tensor)

    result -= einsum('ijkl->jikl', tensor)
    result += einsum('ijkl->jilk', tensor)
    result -= einsum('ijkl->jlik', tensor)
    result += einsum('ijkl->jlki', tensor)
    result -= einsum('ijkl->jkli', tensor)
    result += einsum('ijkl->jkil', tensor)

    result -= einsum('ijkl->kjil', tensor)
    result += einsum('ijkl->kjli', tensor)
    result -= einsum('ijkl->klji', tensor)
    result += einsum('ijkl->klij', tensor)
    result -= einsum('ijkl->kilj', tensor)
    result += einsum('ijkl->kijl', tensor)

    result -= einsum('ijkl->ljki', tensor)
    result += einsum('ijkl->ljik', tensor)
    result -= einsum('ijkl->lijk', tensor)
    result += einsum('ijkl->likj', tensor)
    result -= einsum('ijkl->lkij', tensor)
    result += einsum('ijkl->lkji', tensor)

    return result / 24.0


# Create initial amplitude tensors

def getInitiators():
    I1 = np.ones((TOTAL_SIZE, TOTAL_SIZE), dtype=float)
    indices = np.indices((TOTAL_SIZE, TOTAL_SIZE))

    I1 *= indices[0] < indices[1]
    I1 *= 1e-1 * 2.0

    I2 = np.ones((TOTAL_SIZE, TOTAL_SIZE, TOTAL_SIZE, TOTAL_SIZE), dtype=float)
    indices = np.indices((TOTAL_SIZE, TOTAL_SIZE, TOTAL_SIZE, TOTAL_SIZE))

    I2 *= indices[0] < indices[1]
    I2 *= indices[1] < indices[2]
    I2 *= indices[2] < indices[3]
    I2 *= 1e-4 * 24.0

    return I1, I2


if __name__ == "__main__":

    tc.set_default_dtype(tc.float32)

    print("Building tensors...")

    # Build or load the potential, kinetic tensors and number operator sign vector

    if os.path.exists('V.bin'):
        V = pickle.load(open('V.bin', 'rb'))
    else:
        V = getPotentialTensor()

        pickle.dump(V, open('V.bin', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    K = getKineticTensor()
    L = getNumberSigns()

    # Allocate the amplitudes

    print("Allocating amplitudes...")

    I1, I2 = getInitiators()

    T1 = tc.tensor(I1, dtype=tc.float, requires_grad=True)
    X1 = tc.tensor(I1, dtype=tc.float, requires_grad=True)
    T2 = tc.tensor(I2, dtype=tc.float, requires_grad=True)
    X2 = tc.tensor(I2, dtype=tc.float, requires_grad=True)

    parameters = [T1, X1, T2, X2]

    # Define the minimization function

    def minimization_function(parameters):

        T1, X1, T2, X2 = parameters

        T2_skew = antisymmetrizeRank4Tensor(T2)
        X2_skew = antisymmetrizeRank4Tensor(X2)
        T1_skew = antisymmetrizeRank2Tensor(T1)
        X1_skew = antisymmetrizeRank2Tensor(X1)

        particle_number = eccsd_particle_number(
            L, T1_skew, X1_skew, T2_skew, X2_skew)
        energy = eccsd_energy(K, V, T1_skew, X1_skew, T2_skew, X2_skew)

        return energy.item(), particle_number.item(), energy + (particle_number - FILLING) ** 2

    # Run optimization

    print("Running simulation...")

    optimizer = tc.optim.SGD(parameters, lr=STEP)

    previous_energy = np.inf
    convergence_data = []

    while True:

        energy, particle_number, variable = minimization_function(parameters)
        optimizer.zero_grad()

        variable.backward()

        optimizer.step()

        # Check for convergence

        energy_change = np.abs(1 - energy / previous_energy)
        print(f"Current energy change is {energy_change}!")

        convergence_data.append([energy, particle_number])

        if energy_change < TOLERANCE:
            break

        previous_energy = energy

    # Save data
    np.savetxt("Convergence Data.csv", np.array(
        convergence_data), delimiter=', ')

    print("Calculating excited states...")

    T2_skew = antisymmetrizeRank4Tensor(T2)
    X2_skew = antisymmetrizeRank4Tensor(X2)
    T1_skew = antisymmetrizeRank2Tensor(T1)
    X1_skew = antisymmetrizeRank2Tensor(X1)

    # Calculate excited energies
    band_data = []

    for a in range(K_POINTS.shape[0]):
        band_a, kx_a, ky_a = getProperties(a)
        b = a + K_POINTS.shape[0]

        matrix = np.zeros((2, 2))

        matrix[0, 0] = eccsd_excitation_energy(
            K, V, T1_skew, X1_skew, T2_skew, X2_skew, a, a)
        matrix[1, 1] = eccsd_excitation_energy(
            K, V, T1_skew, X1_skew, T2_skew, X2_skew, b, b)
        matrix[1, 0] = eccsd_excitation_energy(
            K, V, T1_skew, X1_skew, T2_skew, X2_skew, b, a)
        matrix[0, 1] = eccsd_excitation_energy(
            K, V, T1_skew, X1_skew, T2_skew, X2_skew, a, b)

        energies = np.linalg.eigvals(matrix)

        band_data.append([kx_a, ky_a, energies[0], energies[1]])

    # Save data
    np.savetxt("Band Data.csv", np.array(band_data), delimiter=', ')
