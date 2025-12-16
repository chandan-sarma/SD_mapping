#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Original data
H = [
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

# Correct reverse index map: keys to tuples, not a comma separated expression
reverse_index_map = {
    0: (1, 9),
    1: (2, 8),
    2: (2, 11),
    3: (3, 7),
    4: (3, 10),
    5: (4, 9),
    6: (5, 8),
    7: (5, 11),
}

# Build new H_new list
H_new = []
for i, j, x in H:
    if i in reverse_index_map and j in reverse_index_map:
        new_i = reverse_index_map[i]
        new_j = reverse_index_map[j]
        # Append (i1, i2, j1, j2, coeff) or similar structure as needed
        H_new.append((new_i[0], new_i[1], new_j[1], new_j[0], x))

# Output result
# print("H_new = [")
# for row in H_new:
#     print(f"    {row},")
# print("]")


# In[2]:


import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Create list of terms from your matrix
terms = {}
n_modes = 12

# Build Fermionic operator terms
for (p, q, r, s, coeff) in H_new:
    label = f"+_{p} +_{q} -_{r} -_{s}"
    if label in terms:
        terms[label] += coeff
    else:
        terms[label] = coeff

# Create FermionicOp
fer_op = FermionicOp(terms, num_spin_orbitals=n_modes)

# Map to qubit operator via JW
mapper = JordanWignerMapper()
H_JW = mapper.map(fer_op)

# Print Pauli decomposition
print(H_JW)
print('')
print('Number of Pauli terms =', len(H_JW))
print('')
print(f"Number of qubits: {H_JW.num_qubits}")

# eigenvalues, eigenstates = np.linalg.eigh(H_JW.to_matrix())
# for i in range(100, 150):
#     print(i, eigenvalues[i])
   


# In[10]:


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

# Initial parameter values
theta = np.array([-1.6700806543977933, -0.8628110757207489, 
                           -4.808960754454428, 1.0372529296117894, 
                           -4.536694097333226, 2.3716382607482744, -0.9817712923543664])


n_qubits = 12

# === Custom double excitation gate ===
from qiskit.circuit import Parameter

# def double_ex(qc, 0, 1, 2, 3, theta):
def double_ex(qc, i, j, k, l, theta):  
    qc.cx(k,l)
    qc.cx(i,k)
    qc.h(i)
    qc.h(l)
    qc.cx(i,j)
    qc.cx(k,l)
    qc.ry(-theta,i)
    qc.ry(theta,j)
    
    qc.cx(i,l)
    qc.h(l)
    qc.cx(l,j)
    qc.ry(-theta,i)
    qc.ry(theta,j)
    
    qc.cx(k,j)
    qc.cx(k,i)
    qc.ry(theta,i)
    qc.ry(-theta,j)
    
    qc.cx(l,j)
    qc.h(l)
    qc.cx(i,l)
    qc.ry(theta,i)
    qc.ry(-theta,j)
    qc.cx(i,j)
    qc.cx(k,i)
    qc.h(i)
    qc.h(l)
    qc.cx(i,k)
    qc.cx(k,l)



"""
    6Li (M = 1)
    ----------------
    0: (1, 9),
    1: (2, 8),
    2: (2, 11),
    3: (3, 7),
    4: (3, 10),
    5: (4, 9),
    6: (5, 8),
    7: (5, 11),


"""

# __________Ansatz design 1________________

#Initial state (Hartree-Fock reference)
init_occup = [1, 9]
excited_states = [
    [2, 8],
    [2, 11],
    [3, 7],
    [3, 10],
    # [4, 9],
    [5, 8],
    [5, 11],
]

qc = QuantumCircuit(n_qubits)
# Prepare Hartree-Fock state
for i in init_occup:
    qc.x(i)

# Add parameterized double excitations
for idx, (k, l) in enumerate(excited_states):
    i, j = init_occup
    double_ex(qc, i, j, k, l, theta[idx])

double_ex(qc, 3, 10, 4, 9, theta[6])


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
