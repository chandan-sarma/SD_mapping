#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Run1 is considering JW Hamiltonain and 0-to-all connecting ansatz

import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Create list of terms from your matrix
terms = {}
n_modes = 15

# Your Hamiltonian matrix entries:
data = [
(0, 0, -5.3685),
(0, 1, -0.0761),
(0, 2, -2.6335),
(0, 3, 0.1575),
(0, 4, -0.0761),
(0, 5, -1.5117),
(0, 6, -1.4295),
(0, 7, 2.5310),
(0, 9, 1.5117),
(0, 10, -0.5765),
(0, 11, -2.5525),
(0, 12, -0.0761),
(0, 13, -0.4243),
(0, 14, 2.0653),
(1, 0, -0.0761),
(1, 1, -10.5055),
(1, 2, 0.1522),
(1, 3, -2.5339),
(1, 4, 3.5709),
(1, 5, 0.2805),
(1, 7, 0.8700),
(1, 8, 1.2432),
(1, 10, -2.0653),
(1, 11, 0.6417),
(1, 12, 1.1194),
(1, 13, 1.5056),
(2, 0, -2.6335),
(2, 1, 0.1522),
(2, 2, -5.9712),
(2, 3, -0.9350),
(2, 4, 1.6639),
(2, 6, 0.2805),
(2, 7, -3.6800),
(2, 8, 0.1364),
(2, 10, 0.0761),
(2, 11, 2.0062),
(2, 12, -0.5004),
(2, 14, -1.5056),
(3, 0, 0.1575),
(3, 1, -2.5339),
(3, 2, -0.9350),
(3, 3, -6.1320),
(3, 4, -1.0222),
(3, 5, 0.6417),
(3, 8, -1.3268),
(3, 9, -0.8700),
(3, 10, 0.4243),
(3, 11, 1.0925),
(3, 13, 0.2721),
(3, 14, -1.1194),
(4, 0, -0.0761),
(4, 1, 3.5709),
(4, 2, 1.6639),
(4, 3, -1.0222),
(4, 4, -2.5677),
(4, 6, -0.6417),
(4, 8, -1.3751),
(4, 9, -3.6800),
(4, 10, 2.5525),
(4, 12, -1.0925),
(4, 13, -2.0062),
(4, 14, -0.8700),
(5, 0, -1.5117),
(5, 1, 0.2805),
(5, 3, 0.6417),
(5, 5, -10.5055),
(5, 6, 1.7400),
(5, 7, 0.8700),
(5, 8, -1.5069),
(5, 9, 3.5709),
(5, 10, 2.6250),
(5, 11, -0.6417),
(5, 13, -0.9459),
(6, 0, -1.4295),
(6, 2, 0.2805),
(6, 4, -0.6417),
(6, 5, 1.7400),
(6, 6, -9.2830),
(6, 7, 1.7100),
(6, 8, -0.8097),
(6, 9, 0.8700),
(6, 10, 1.5117),
(6, 12, -0.6417),
(6, 14, 0.9459),
(7, 0, 2.5310),
(7, 1, 0.8700),
(7, 2, -3.6800),
(7, 5, 0.8700),
(7, 6, 1.7100),
(7, 7, -5.3225),
(7, 8, -0.8097),
(7, 9, 0.2283),
(7, 10, -1.5117),
(7, 11, -1.4600),
(7, 12, 0.8700),
(8, 1, 1.2432),
(8, 2, 0.1364),
(8, 3, -1.3268),
(8, 4, -1.3751),
(8, 5, -1.5069),
(8, 6, -0.8097),
(8, 7, -0.8097),
(8, 8, -7.0612),
(8, 9, -1.5069),
(8, 11, -0.9461),
(8, 12, 0.7348),
(8, 13, 0.1318),
(8, 14, 0.9695),
(9, 0, 1.5117),
(9, 3, -0.8700),
(9, 4, -3.6800),
(9, 5, 3.5709),
(9, 6, 0.8700),
(9, 7, 0.2283),
(9, 8, -1.5069),
(9, 9, -2.5677),
(9, 10, -3.0987),
(9, 13, 1.4600),
(9, 14, 0.8700),
(10, 0, -0.5765),
(10, 1, -2.0653),
(10, 2, 0.0761),
(10, 3, 0.4243),
(10, 4, 2.5525),
(10, 5, 2.6250),
(10, 6, 1.5117),
(10, 7, -1.5117),
(10, 9, -3.0987),
(10, 10, -2.2452),
(10, 11, -0.4243),
(10, 12, 0.1575),
(10, 13, -2.2650),
(10, 14, -0.4243),
(11, 0, -2.5525),
(11, 1, 0.6417),
(11, 2, 2.0062),
(11, 3, 1.0925),
(11, 5, -0.6417),
(11, 7, -1.4600),
(11, 8, -0.9461),
(11, 10, -0.4243),
(11, 11, -3.6702),
(11, 12, 1.2943),
(11, 13, -2.1643),
(11, 14, 3.5709),
(12, 0, -0.0761),
(12, 1, 1.1194),
(12, 2, -0.5004),
(12, 4, -1.0925),
(12, 6, -0.6417),
(12, 7, 0.8700),
(12, 8, 0.7348),
(12, 10, 0.1575),
(12, 11, 1.2943),
(12, 12, -3.8065),
(12, 13, -0.9350),
(12, 14, 1.2943),
(13, 0, -0.4243),
(13, 1, 1.5056),
(13, 3, 0.2721),
(13, 4, -2.0062),
(13, 5, -0.9459),
(13, 8, 0.1318),
(13, 9, 1.4600),
(13, 10, -2.2650),
(13, 11, -2.1643),
(13, 12, -0.9350),
(13, 13, -0.0602),
(13, 14, -2.1643),
(14, 0, 2.0653),
(14, 2, -1.5056),
(14, 3, -1.1194),
(14, 4, -0.8700),
(14, 6, 0.9459),
(14, 8, 0.9695),
(14, 9, 0.8700),
(14, 10, -0.4243),
(14, 11, 3.5709),
(14, 12, 1.2943),
(14, 13, -2.1643),
(14, 14, -1.1665),
]

from qiskit.quantum_info import SparsePauliOp

num_qubits = 15

# Merge symmetric terms
pair_terms = {}
for i, j, x in data:
    key = tuple(sorted((i, j)))
    pair_terms[key] = pair_terms.get(key, 0) + x

paulis = []
coeffs = []

identity_coeff = 0.0  # To collect total weight of I terms

for (i, j), x in pair_terms.items():
    if i == j:
        # On-site term: (x / 2)(I - Z_i)
        identity_coeff += x / 2  # Collect I term

        z_term = ['I'] * num_qubits
        z_term[i] = 'Z'
        paulis.append(''.join(z_term)[::-1])
        coeffs.append(-x / 2)
    else:
        # Ensure i < j for consistency
        if i > j:
            i, j = j, i

        # Off-diagonal term: (x / 2)(X_i Z..Z X_j + Y_i Z..Z Y_j)
        # Each term gets x / 4 due to symmetrization
        z_string = ['I'] * num_qubits
        for k in range(i + 1, j):
            z_string[k] = 'I'

        for op in ['X', 'Y']:
            jw_term = z_string.copy()
            jw_term[i] = op
            jw_term[j] = op
            paulis.append(''.join(jw_term)[::-1])
            coeffs.append(x / 4)

# After the loop, add the combined identity term once (if non-zero)
if abs(identity_coeff) > 1e-12:  # Avoid tiny noise
    paulis.append('I' * num_qubits)
    coeffs.append(identity_coeff)

# Build SparsePauliOp
H_JW = SparsePauliOp.from_list(list(zip(paulis, coeffs)))

# Print Hamiltonian
#print(H_JW)
print('')
print('Number of Pauli terms =', len(H_JW))


# In[33]:


#QiskitRuntimeService(channel = 'local')

from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit import QuantumCircuit, transpile
 
service = QiskitRuntimeService(channel = 'local')
backend = FakeFez()

ref_value = -14.926144966493986

params = np.array([-1.7683014299738231, 5.323049349543284, 4.826306237532618, 
                   4.978449150096527, 1.8362980656240089, 0.7128385409982549, 
                   -4.5178055792008465, -4.484873199572197, -1.972774028141386, 
                   1.078555440705402, 3.8676636270906504, -7.050261076083411, 
                   -1.093262696500093, 6.937135865846558])

qc = QuantumCircuit(15)

# Initial gates
qc.x(0)
qc.h(0)
qc.cx(0, 1)
qc.ry(-params[0], 0)
qc.ry(-params[0], 1)
qc.cx(0, 1)
qc.h(0)

# Loop over qubits 2, 8
for i in range(2, 15):
    qc.h(i-1)
    qc.cx(i-1, i)
    qc.ry(-params[i - 1], i-1)
    qc.ry(-params[i - 1], i)
    qc.cx(i-1, i)
    qc.h(i-1)
    
pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
isa_psi = pm.run(qc)
isa_observables = H_JW.apply_layout(isa_psi.layout)

print("No. of qubits:", isa_psi.num_qubits)
print("No. of gates:", isa_psi.count_ops())
print("Depth:", isa_psi.depth())
 
estimator = Estimator(mode=backend)
 
# --- Estimator Loop ---
runs = []
eigenvalues = []
for run in range(100):
    job = estimator.run([(isa_psi, isa_observables)])
    result = job.result()
    pub_result = result[0]
    eigenvalue = float(pub_result.data.evs)
    relative_error = abs((eigenvalue - ref_value) / ref_value)*100

    runs.append(run)
    eigenvalues.append(eigenvalue)

    # Print with comma-separated values: run, eigenvalue, relative_error
    print(f"({run},{eigenvalue},{relative_error}),")

print('')
print('Noise level = 3')
# Create an empty circuit with the same qubits and classical bits
modified_circuit1 = QuantumCircuit(*isa_psi.qregs, *isa_psi.cregs)

# Loop over the instructions in the transpiled circuit
for instr, qargs, cargs in isa_psi.data:
    if instr.name == 'cz':
        for _ in range(3):
            modified_circuit1.append(instr.copy(), qargs, cargs)
    else:
        modified_circuit1.append(instr.copy(), qargs, cargs)    

print(f'Qubit no. for Noise = 3 circuit: {modified_circuit1.num_qubits}, Qubit no. for optimized Hamil :{isa_observables.num_qubits}')
print('')
print("Gate counts of optimized circuit:", modified_circuit1.count_ops())
print("Depth of optimized circuit:", modified_circuit1.depth())
# transpiled_final.draw('mpl', style='iqp')

## --- Estimator Loop ---
runs = []
eigenvalues = []
for run in range(100):
    job = estimator.run([(modified_circuit1, isa_observables)])
    result = job.result()
    pub_result = result[0]
    eigenvalue = float(pub_result.data.evs)
    relative_error = abs((eigenvalue - ref_value) / ref_value)*100

    runs.append(run)
    eigenvalues.append(eigenvalue)

    # Print with comma-separated values: run, eigenvalue, relative_error
    print(f"({run},{eigenvalue},{relative_error}),")   


print('')
print('Noise level = 5')
# Create an empty circuit with the same qubits and classical bits
modified_circuit2 = QuantumCircuit(*isa_psi.qregs, *isa_psi.cregs)

# Loop over the instructions in the transpiled circuit
for instr, qargs, cargs in isa_psi.data:
    if instr.name == 'cz':
        for _ in range(5):
            modified_circuit2.append(instr.copy(), qargs, cargs)
    else:
        modified_circuit2.append(instr.copy(), qargs, cargs)    

print(f'Qubit no. for Noise = 5 circuit: {modified_circuit2.num_qubits}, Qubit no. for optimized Hamil :{isa_observables.num_qubits}')
print('')
print("Gate counts of optimized circuit:", modified_circuit2.count_ops())
print("Depth of optimized circuit:", modified_circuit2.depth())
# transpiled_final.draw('mpl', style='iqp')

## --- Estimator Loop ---
runs = []
eigenvalues = []
for run in range(100):
    job = estimator.run([(modified_circuit2, isa_observables)])
    result = job.result()
    pub_result = result[0]
    eigenvalue = float(pub_result.data.evs)
    relative_error = abs((eigenvalue - ref_value) / ref_value)*100

    runs.append(run)
    eigenvalues.append(eigenvalue)

    # Print with comma-separated values: run, eigenvalue, relative_error
    print(f"({run},{eigenvalue},{relative_error}),")       
