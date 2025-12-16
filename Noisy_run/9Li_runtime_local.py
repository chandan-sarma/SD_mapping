#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Run1 is considering JW Hamiltonain and 0-to-all connecting ansatz

import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Create list of terms from your matrix
terms = {}
n_modes = 15

# Your Hamiltonian matrix entries:
data = [
(0, 0, -11.7125),
(0, 1, -2.4578),
(0, 2, -1.3268),
(0, 3, 0.1575),
(0, 4, -0.0761),
(0, 5, 1.5117),
(0, 6, 1.4295),
(0, 9, 1.5117),
(0, 10, -1.2943),
(0, 11, -1.1194),
(0, 12, 1.2432),
(0, 13, -0.4243),
(0, 14, 2.0653),
(1, 0, -2.4578),
(1, 1, -5.3307),
(1, 2, -0.1318),
(1, 3, 0.0761),
(1, 4, -2.6335),
(1, 7, 1.5117),
(1, 8, -1.4295),
(1, 9, 2.5310),
(1, 10, -2.0062),
(1, 11, -0.1522),
(1, 12, 0.9461),
(1, 13, 2.5525),
(1, 14, -0.0761),
(2, 0, -1.3268),
(2, 1, -0.1318),
(2, 2, -10.5027),
(2, 3, -0.8097),
(2, 4, -1.5069),
(2, 5, -1.3751),
(2, 6, -1.3268),
(2, 7, 0.1364),
(2, 8, 1.2432),
(2, 10, 1.3751),
(2, 11, 0.9695),
(2, 12, -1.2182),
(3, 0, 0.1575),
(3, 1, 0.0761),
(3, 2, -0.8097),
(3, 3, -11.4590),
(3, 4, -0.1522),
(3, 5, -0.6417),
(3, 7, 0.2805),
(3, 9, -0.8700),
(3, 10, 0.4243),
(3, 11, 2.0653),
(3, 13, 1.0660),
(3, 14, -1.1194),
(4, 0, -0.0761),
(4, 1, -2.6335),
(4, 2, -1.5069),
(4, 3, -0.1522),
(4, 4, -6.4797),
(4, 6, 0.6417),
(4, 8, 0.2805),
(4, 9, -3.6800),
(4, 10, 2.5525),
(4, 11, 0.0761),
(4, 13, -2.0062),
(4, 14, -0.0761),
(5, 0, 1.5117),
(5, 2, -1.3751),
(5, 3, -0.6417),
(5, 5, -13.2575),
(5, 6, 0.8700),
(5, 7, 0.6417),
(5, 8, 3.5709),
(5, 9, -3.5709),
(5, 10, -2.6250),
(5, 12, 0.9695),
(5, 13, 0.9459),
(6, 0, 1.4295),
(6, 2, -1.3268),
(6, 4, 0.6417),
(6, 5, 0.8700),
(6, 6, -10.6200),
(6, 7, -0.9350),
(6, 8, -1.5117),
(6, 9, -0.8700),
(6, 10, -1.5117),
(6, 12, 0.1318),
(6, 14, -0.9459),
(7, 1, 1.5117),
(7, 2, 0.1364),
(7, 3, 0.2805),
(7, 5, 0.6417),
(7, 6, -0.9350),
(7, 7, -12.0052),
(7, 8, -1.7400),
(7, 9, -0.8700),
(7, 11, -2.6250),
(7, 12, 0.7348),
(7, 13, -0.6417),
(8, 1, -1.4295),
(8, 2, 1.2432),
(8, 4, 0.2805),
(8, 5, 3.5709),
(8, 6, -1.5117),
(8, 7, -1.7400),
(8, 8, -10.3377),
(8, 9, 1.7100),
(8, 11, 1.5117),
(8, 12, -0.9461),
(8, 14, -0.6417),
(9, 0, 1.5117),
(9, 1, 2.5310),
(9, 3, -0.8700),
(9, 4, -3.6800),
(9, 5, -3.5709),
(9, 6, -0.8700),
(9, 7, -0.8700),
(9, 8, 1.7100),
(9, 9, -6.3772),
(9, 10, -3.0987),
(9, 11, -1.5117),
(9, 13, 1.4600),
(9, 14, 0.8700),
(10, 0, -1.2943),
(10, 1, -2.0062),
(10, 2, 1.3751),
(10, 3, 0.4243),
(10, 4, 2.5525),
(10, 5, -2.6250),
(10, 6, -1.5117),
(10, 9, -3.0987),
(10, 10, -5.5299),
(10, 11, -1.3157),
(10, 12, 0.1364),
(10, 13, -2.2650),
(10, 14, -0.4243),
(11, 0, -1.1194),
(11, 1, -0.1522),
(11, 2, 0.9695),
(11, 3, 2.0653),
(11, 4, 0.0761),
(11, 7, -2.6250),
(11, 8, 1.5117),
(11, 9, -1.5117),
(11, 10, -1.3157),
(11, 11, -3.8187),
(11, 12, -0.7348),
(11, 13, 0.4243),
(11, 14, 0.1575),
(12, 0, 1.2432),
(12, 1, 0.9461),
(12, 2, -1.2182),
(12, 5, 0.9695),
(12, 6, 0.1318),
(12, 7, 0.7348),
(12, 8, -0.9461),
(12, 10, 0.1364),
(12, 11, -0.7348),
(12, 12, -7.1699),
(12, 13, -0.8097),
(12, 14, -1.5069),
(13, 0, -0.4243),
(13, 1, 2.5525),
(13, 3, 1.0660),
(13, 4, -2.0062),
(13, 5, 0.9459),
(13, 7, -0.6417),
(13, 9, 1.4600),
(13, 10, -2.2650),
(13, 11, 0.4243),
(13, 12, -0.8097),
(13, 13, -3.8912),
(13, 14, -1.2943),
(14, 0, 2.0653),
(14, 1, -0.0761),
(14, 3, -1.1194),
(14, 4, -0.0761),
(14, 6, -0.9459),
(14, 8, -0.6417),
(14, 9, 0.8700),
(14, 10, -0.4243),
(14, 11, 0.1575),
(14, 12, -1.5069),
(14, 13, -1.2943),
(14, 14, -3.5825),
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

ref_value = -18.90629232811285

params = np.array([-1.770004578672819, -1.6712686607273812, 4.86015809804485, 
                   4.832224502090953, -1.4790332106307376, -0.714376008433817, 
                   1.655293289887329, -4.897148718251804, 2.40699246818179, 
                   -2.73835206657126, -2.89158628801402, -2.827540973039091, 
                   -4.1077074598509205, -2.830228504387003])

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
modified_circuit1 = QuantumCircuit(*isa_psi.qregs, *isa_psi.cregs)

# Loop over the instructions in the transpiled circuit
for instr, qargs, cargs in isa_psi.data:
    if instr.name == 'cz':
        for _ in range(5):
            modified_circuit1.append(instr.copy(), qargs, cargs)
    else:
        modified_circuit1.append(instr.copy(), qargs, cargs)    

print(f'Qubit no. for Noise = 5 circuit: {modified_circuit.num_qubits}, Qubit no. for optimized Hamil :{isa_observables.num_qubits}')
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
