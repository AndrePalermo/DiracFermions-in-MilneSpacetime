import mpmath as mp
import numpy as np
import scipy.integrate as spi
from mpmath import pauli_matrices, Adag_A, trace, to_numpy
from joblib import Parallel, delayed

# Set precision
mp.dps = 800

fmToGeV = 5.067730474  # 1fm = 5.067730474 GeV^-1

"""
    Saves to file the table:
        SP[adimensional] pT[GeV] T[GeV] tau[GeV^-1] num_Pz den_Pz
    polarization is then num_Pz/den_Pz
"""

def general_num(mu, pT, SP_val, T):
    return np.real(to_numpy(trace(pauli_matrices(3) * Adag_A(τ, mu, mass, pT, 0, 1/T, SP_val)) / 2))

def general_den(mu, pT, SP_val, T):
    return np.real(to_numpy(trace(pauli_matrices(0) * Adag_A(τ, mu, mass, pT, 0, 1/T, SP_val)) / 2))

def process_combination(s, pT, T):
    with mp.workprec(800):
        resultden, error = spi.quad(lambda mu: general_den(mu, pT, s, T), -150, 150)
        resultnum, nerror = spi.quad(lambda mu: general_num(mu, pT, s, T), -150, 150)
    return (s, pT, T, τ, resultnum, resultden)

mass = 1.1  # GeV
pTlist = [0.1, 0.5, 1, 3]  # GeV
τ = 8 * fmToGeV  # GeV^-1
Tlist = [0.150, 0.2, 0.3]  # GeV
SPlist = np.linspace(-5, -0.1, 20)

# Parallel execution
results = Parallel(n_jobs=5)(
    delayed(process_combination)(s, pT, T)
    for s in SPlist
    for pT in pTlist
    for T in Tlist
)

with open('polarization_table.txt', 'w') as f:
    f.write("SP[adimensional]\tpT[GeV]\tT[GeV]\ttau[GeV^-1]\tnum_Pz\tden_Pz\n")
    for res in results:
        f.write(f"{res[0]:.6f}\t{res[1]:.6f}\t{res[2]:.6f}\t{res[3]:.6f}\t{res[4]:.8e}\t{res[6]:.8e}\n")


