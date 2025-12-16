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
(0, 0, -3.2175),
(0, 1, 0.9461),
(0, 2, -2.6335),
(0, 3, 0.1318),
(0, 4, -0.0761),
(0, 5, -1.4295),
(0, 6, 2.5310),
(0, 8, 1.5117),
(0, 10, -0.1522),
(0, 11, -1.5056),
(0, 12, -0.0761),
(0, 13, 0.9695),
(0, 14, 2.0653),
(1, 0, 0.9461),
(1, 1, -1.6615),
(1, 2, 0.0761),
(1, 3, -1.3268),
(1, 4, 0.1575),
(1, 5, 1.5117),
(1, 7, -2.5310),
(1, 9, 1.5117),
(1, 10, -1.0925),
(1, 11, 0.2174),
(1, 12, 2.5525),
(1, 13, -1.3751),
(1, 14, -0.4243),
(2, 0, -2.6335),
(2, 1, 0.0761),
(2, 2, -4.3665),
(2, 3, 1.5069),
(2, 4, 1.6639),
(2, 5, 0.2805),
(2, 6, -3.6800),
(2, 7, 0.8700),
(2, 10, 0.0761),
(2, 11, 2.0653),
(2, 12, -0.0761),
(2, 14, -1.5056),
(3, 0, 0.1318),
(3, 1, -1.3268),
(3, 2, 1.5069),
(3, 3, -4.4122),
(3, 4, -0.8097),
(3, 6, 1.3751),
(3, 7, -1.3268),
(3, 8, 0.1364),
(3, 9, -1.2432),
(3, 10, 0.9461),
(3, 11, -1.2432),
(3, 13, 0.2935),
(4, 0, -0.0761),
(4, 1, 0.1575),
(4, 2, 1.6639),
(4, 3, -0.8097),
(4, 4, -1.4080),
(4, 5, -0.6417),
(4, 8, -3.6800),
(4, 9, -0.8700),
(4, 10, 2.5525),
(4, 11, 0.4243),
(4, 12, -1.0925),
(4, 14, -0.4457),
(5, 0, -1.4295),
(5, 1, 1.5117),
(5, 2, 0.2805),
(5, 4, -0.6417),
(5, 5, -8.2245),
(5, 6, 1.7100),
(5, 7, 0.8700),
(5, 8, 0.8700),
(5, 9, -3.5709),
(5, 10, 1.5117),
(5, 11, -2.6250),
(5, 12, -0.6417),
(5, 14, 0.9459),
(6, 0, 2.5310),
(6, 2, -3.6800),
(6, 3, 1.3751),
(6, 5, 1.7100),
(6, 6, -4.2640),
(6, 8, 0.2283),
(6, 9, 3.5709),
(6, 10, -1.5117),
(6, 12, 0.8700),
(6, 13, -0.9695),
(7, 1, -2.5310),
(7, 2, 0.8700),
(7, 3, -1.3268),
(7, 5, 0.8700),
(7, 7, -0.5690),
(7, 8, -0.9350),
(7, 9, 0.6417),
(7, 11, 1.5117),
(7, 12, -1.4600),
(7, 13, 0.1318),
(8, 0, 1.5117),
(8, 3, 0.1364),
(8, 4, -3.6800),
(8, 5, 0.8700),
(8, 6, 0.2283),
(8, 7, -0.9350),
(8, 8, -1.9542),
(8, 9, 0.8700),
(8, 10, -3.0987),
(8, 13, 0.7348),
(8, 14, 0.8700),
(9, 1, 1.5117),
(9, 3, -1.2432),
(9, 4, -0.8700),
(9, 5, -3.5709),
(9, 6, 3.5709),
(9, 7, 0.6417),
(9, 8, 0.8700),
(9, 9, 0.7708),
(9, 11, -3.0987),
(9, 13, 0.9461),
(9, 14, 1.4600),
(10, 0, -0.1522),
(10, 1, -1.0925),
(10, 2, 0.0761),
(10, 3, 0.9461),
(10, 4, 2.5525),
(10, 5, 1.5117),
(10, 6, -1.5117),
(10, 8, -3.0987),
(10, 10, -0.8267),
(10, 11, 1.3157),
(10, 12, 0.1575),
(10, 13, 0.7348),
(10, 14, -0.4243),
(11, 0, -1.5056),
(11, 1, 0.2174),
(11, 2, 2.0653),
(11, 3, -1.2432),
(11, 4, 0.4243),
(11, 5, -2.6250),
(11, 7, 1.5117),
(11, 9, -3.0987),
(11, 10, 1.3157),
(11, 11, -0.0342),
(11, 12, 0.4243),
(11, 13, 0.1364),
(11, 14, -2.2650),
(12, 0, -0.0761),
(12, 1, 2.5525),
(12, 2, -0.0761),
(12, 4, -1.0925),
(12, 5, -0.6417),
(12, 6, 0.8700),
(12, 7, -1.4600),
(12, 10, 0.1575),
(12, 11, 0.4243),
(12, 12, -0.5905),
(12, 13, 1.5069),
(12, 14, 1.2943),
(13, 0, 0.9695),
(13, 1, -1.3751),
(13, 3, 0.2935),
(13, 6, -0.9695),
(13, 7, 0.1318),
(13, 8, 0.7348),
(13, 9, 0.9461),
(13, 10, 0.7348),
(13, 11, 0.1364),
(13, 12, 1.5069),
(13, 13, -1.6742),
(13, 14, -0.8097),
(14, 0, 2.0653),
(14, 1, -0.4243),
(14, 2, -1.5056),
(14, 4, -0.4457),
(14, 5, 0.9459),
(14, 8, 0.8700),
(14, 9, 1.4600),
(14, 10, -0.4243),
(14, 11, -2.2650),
(14, 12, 1.2943),
(14, 13, -0.8097),
(14, 14, 1.6045),
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
print(H_JW)
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

ref_value = -14.60679026967584

params = np.array([-4.922575964816499, 4.902209499458138, -1.3520889505992117, 
                  -1.7371552697831525, -4.6030804043061835, -0.8228501543235343, 
                  4.046656596151176, -1.7309287734163423, 4.518670119298085, 
                  5.4758225992248395, -4.138720139171718, -3.7100606352959695, 
                  5.355182660245251, 2.3612419825440316])

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
    
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
isa_psi = pm.run(qc)
isa_observables = H_JW.apply_layout(isa_psi.layout)

print("No. of qubits:", isa_psi.num_qubits)
print("No. of gates:", isa_psi.count_ops())
print("Depth:", isa_psi.depth())
 
estimator = Estimator(mode=backend)
 
# calculate [ <psi(theta1)|hamiltonian|psi(theta)> ]
job = estimator.run([(isa_psi, isa_observables)])
pub_result = job.result()[0]
print(f"Expectation values: {pub_result.data.evs}")
print('')
# print(psi.decompose())
# print('')
# print(isa_psi)

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
modified_circuit = QuantumCircuit(*isa_psi.qregs, *isa_psi.cregs)

# Loop over the instructions in the transpiled circuit
for instr, qargs, cargs in isa_psi.data:
    if instr.name == 'cz':
        for _ in range(3):
            modified_circuit.append(instr.copy(), qargs, cargs)
    else:
        modified_circuit.append(instr.copy(), qargs, cargs)    

print(f'Qubit no. for Noise = 3 circuit: {modified_circuit.num_qubits}, Qubit no. for optimized Hamil :{isa_observables.num_qubits}')
print('')
print("Gate counts of optimized circuit:", modified_circuit.count_ops())
print("Depth of optimized circuit:", modified_circuit.depth())
# transpiled_final.draw('mpl', style='iqp')

## --- Estimator Loop ---
runs = []
eigenvalues = []
for run in range(100):
    job = estimator.run([(modified_circuit, isa_observables)])
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
modified_circuit = QuantumCircuit(*isa_psi.qregs, *isa_psi.cregs)

# Loop over the instructions in the transpiled circuit
for instr, qargs, cargs in isa_psi.data:
    if instr.name == 'cz':
        for _ in range(5):
            modified_circuit.append(instr.copy(), qargs, cargs)
    else:
        modified_circuit.append(instr.copy(), qargs, cargs)    

print(f'Qubit no. for Noise = 5 circuit: {modified_circuit.num_qubits}, Qubit no. for optimized Hamil :{isa_observables.num_qubits}')
print('')
print("Gate counts of optimized circuit:", modified_circuit.count_ops())
print("Depth of optimized circuit:", modified_circuit.depth())
# transpiled_final.draw('mpl', style='iqp')

## --- Estimator Loop ---
runs = []
eigenvalues = []
for run in range(100):
    job = estimator.run([(modified_circuit, isa_observables)])
    result = job.result()
    pub_result = result[0]
    eigenvalue = float(pub_result.data.evs)
    relative_error = abs((eigenvalue - ref_value) / ref_value)*100

    runs.append(run)
    eigenvalues.append(eigenvalue)

    # Print with comma-separated values: run, eigenvalue, relative_error
    print(f"({run},{eigenvalue},{relative_error}),")       

