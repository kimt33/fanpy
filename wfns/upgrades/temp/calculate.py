import numpy as np
import os
from wfns.wfn.geminal.apig import APIG
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.backend.sd_list import sd_list
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy
from wfns.solver.equation import minimize
import sys
sys.path.append('../')
import speedup_sd, speedup_sign
import speedup_apg
import speedup_objective


# Number of electrons
nelec = 10
print('Number of Electrons: {}'.format(nelec))

# Number of spin orbitals
nspin = 20
print('Number of Spin Orbitals: {}'.format(nspin))

# One-electron integrals
one_int_file = 'oneint.npy'
one_int = np.load(one_int_file)
print('One-Electron Integrals: {}'.format(os.path.abspath(one_int_file)))

# Two-electron integrals
two_int_file = 'twoint.npy'
two_int = np.load(two_int_file)
print('Two-Electron Integrals: {}'.format(os.path.abspath(two_int_file)))

# Nuclear-nuclear repulsion
nuc_nuc = 20.415320808816475
print('Nuclear-nuclear repulsion: {}'.format(nuc_nuc))

# Initialize wavefunction
wfn = APIG(nelec, nspin, params=None, memory=None, ngem=None)
print('Wavefunction: APIG')

# Initialize Hamiltonian
ham = RestrictedChemicalHamiltonian(one_int, two_int, energy_nuc_nuc=nuc_nuc, params=None,
                                    update_prev_params=True)
print('Hamiltonian: RestrictedChemicalHamiltonian')

# Projection space
pspace = sd_list(nelec, nspin//2, num_limit=None, exc_orders=[2, 4, 6, 8, 10], spin=None,
                 seniority=wfn.seniority)
print('Projection space (orders of excitations): [2, 4, 6, 8, 10]')

# Select parameters that will be optimized
param_selection = [(wfn, np.ones(wfn.nparams, dtype=bool)), (ham, np.ones(ham.nparams, dtype=bool))]

# Initialize objective
objective = OneSidedEnergy(wfn, ham, param_selection=param_selection, tmpfile='checkpoint.npy',
                           refwfn=pspace)
import time
time1 = time.time()
x = objective.objective(objective.params)
y = objective.gradient(objective.params)
print(time.time() - time1)
print(x)
print(y)
import sys
sys.exit()

objective.print_energy = True

# Normalize
wfn.normalize(pspace)

# Solve
print('Optimizing wavefunction: minimize solver')
results = minimize(objective, save_file='checkpoint.npy', method='BFGS', jac=objective.gradient,
                   options={'gtol': 1e-8, 'disp':True})

# Results
if results['success']:
    print('Optimization was successful')
else:
    print('Optimization was not successful: {}'.format(results['message']))
print('Final Electronic Energy: {}'.format(results['energy']))
print('Final Total Energy: {}'.format(results['energy'] + nuc_nuc))

# Save results
if results['success']:
    np.save('hamiltonian.npy', ham.params)
    np.save('hamiltonian_um.npy', ham._prev_unitary)
    np.save('wavefunction.npy', wfn.params)
