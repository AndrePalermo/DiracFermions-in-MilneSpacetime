import mpmath as mp
import numpy as np
import scipy.integrate as spi
from libMilne import pauli_matrices, Adag_A, trace, to_numpy
from joblib import Parallel, delayed

fmToGeV = 5.067730474
print("working")

mass = 1.1       # GeV
pTlist = [0.1, 0.5, 1, 3]   # GeV
τ = 8 * fmToGeV  # GeV^-1
Tlist = [0.150, 0.2, 0.3]   # GeV
SPlist = np.linspace(-5, -0.1, 20)

def general_num(mu, pT, SP_val, T):
    return np.real(to_numpy(trace(pauli_matrices(3) * Adag_A(τ, mu, mass, pT, 0, 1/T, SP_val)) / 2))

def general_den(mu, pT, SP_val, T):
    return np.real(to_numpy(trace(pauli_matrices(0) * Adag_A(τ, mu, mass, pT, 0, 1/T, SP_val)) / 2))

def process_combination(s, pT, T):
    # Set precision inside the worker — avoids pickling mpmath context
    mp.mp.dps = 800
    with mp.workprec(800):
        resultden, _ = spi.quad(lambda mu: general_den(mu, pT, s, T), -150, 150)
        resultnum, _ = spi.quad(lambda mu: general_num(mu, pT, s, T), -150, 150)
    return (s, pT, T, τ, resultnum, resultden)

# Use 'loky' but with no mpmath state to pickle — all mpmath calls are inside the worker now
results = Parallel(n_jobs=5, backend="loky")(
    delayed(process_combination)(s, pT, T)
    for s in SPlist
    for pT in pTlist
    for T in Tlist
)

with open('polarization_table.txt', 'w') as f:
    f.write("SP[adimensional]\tpT[GeV]\tT[GeV]\ttau[GeV^-1]\tnum_Pz\tden_Pz\n")
    for res in results:
        # Fixed: was res[6] (index out of range), should be res[5]
        f.write(f"{res[0]:.6f}\t{res[1]:.6f}\t{res[2]:.6f}\t{res[3]:.6f}\t{res[4]:.8e}\t{res[5]:.8e}\n")