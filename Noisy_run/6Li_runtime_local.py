#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Run1 is considering JW Hamiltonain and 0-to-all connecting ansatz

import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Create list of terms from your matrix
terms = {}
n_modes = 8

# Your Hamiltonian matrix entries:
data = [
(0, 0, -0.1830),
(0, 1, -1.3268),
(0, 2, 0.1318),
(0, 3, -2.5310),
(0, 4, 1.5117),
(0, 5, 0.6417),
(0, 6, -1.3751),
(0, 7, 0.9695),
(1, 0, -1.3268),
(1, 1, -3.4800),
(1, 2, 0.7178),
(1, 3, -1.3268),
(1, 4, -1.2432),
(1, 5, -1.2432),
(1, 6, 0.7178),
(1, 7, -1.1194),
(2, 0, 0.1318),
(2, 1, 0.7178),
(2, 2, 0.8693),
(2, 3, -1.3751),
(2, 4, 0.1364),
(2, 5, 0.9461),
(2, 6, -2.0062),
(2, 7, -0.4243),
(3, 0, -2.5310),
(3, 1, -1.3268),
(3, 2, -1.3751),
(3, 3, -0.1830),
(3, 4, 0.6417),
(3, 5, 1.5117),
(3, 6, 0.1318),
(3, 7, 0.9695),
(4, 0, 1.5117),
(4, 1, -1.2432),
(4, 2, 0.1364),
(4, 3, 0.6417),
(4, 4, 0.7118),
(4, 5, -3.0987),
(4, 6, 0.9461),
(4, 7, 0.7348),
(5, 0, 0.6417),
(5, 1, -1.2432),
(5, 2, 0.9461),
(5, 3, 1.5117),
(5, 4, -3.0987),
(5, 5, 0.7118),
(5, 6, 0.1364),
(5, 7, 0.7348),
(6, 0, -1.3751),
(6, 1, 0.7178),
(6, 2, -2.0062),
(6, 3, 0.1318),
(6, 4, 0.9461),
(6, 5, 0.1364),
(6, 6, 0.8693),
(6, 7, -0.4243),
(7, 0, 0.9695),
(7, 1, -1.1194),
(7, 2, -0.4243),
(7, 3, 0.9695),
(7, 4, 0.7348),
(7, 5, 0.7348),
(7, 6, -0.4243),
(7, 7, 0.5480),

]

from qiskit.quantum_info import SparsePauliOp

num_qubits = 8

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

ref_value = -5.43703843

params = np.array([
    -4.37312990679407, 4.292747747429345,
    -1.2387590748056694, -7.432875204069434,
    3.998985694702535, -2.616950530986742, -6.2103076252179905])

qc = QuantumCircuit(8)
qc.x(0)
qc.h(0)
qc.cx(0, 1)
qc.ry(-params[0], 0)
qc.ry(-params[0], 1)
qc.cx(0, 1)
qc.h(0)

# Loop over qubits 2, 8
for i in range(2, 8):
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
