#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time

start_time = time.time()

import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Create list of terms from your matrix
terms = {}
n_modes = 22

# Your Hamiltonian matrix entries:
data = [
(0, 0, -7.4842),
(0, 1, 0.1360),
(0, 2, -0.0771),
(0, 3, 0.0403),
(0, 4, -0.0198),
(0, 5, -0.2037),
(0, 6, 0.1139),
(0, 7, -0.1037),
(0, 8, 0.0975),
(0, 9, -0.2022),
(0, 10, 0.0591),
(0, 11, -0.0512),
(0, 12, -0.1370),
(0, 13, 0.0737),
(0, 14, -0.0994),
(0, 15, 0.3873),
(0, 16, -0.2645),
(0, 17, 0.2152),
(0, 18, -0.1933),
(0, 19, 0.1818),
(0, 20, -0.1756),
(0, 21, 0.1728),
(1, 0, 0.1360),
(1, 1, -7.3796),
(1, 2, 0.0957),
(1, 3, -0.0718),
(1, 4, 0.0743),
(1, 5, 0.1338),
(1, 6, -0.1703),
(1, 7, 0.1068),
(1, 8, -0.1079),
(1, 9, 0.1227),
(1, 10, -0.1315),
(1, 11, 0.0583),
(1, 12, 0.1159),
(1, 13, -0.0948),
(1, 14, 0.0994),
(1, 15, -0.2098),
(1, 16, 0.3106),
(1, 17, -0.2660),
(1, 18, 0.2230),
(1, 19, -0.2015),
(1, 20, 0.1919),
(1, 21, -0.1877),
(2, 0, -0.0771),
(2, 1, 0.0957),
(2, 2, -7.3588),
(2, 3, 0.1246),
(2, 4, -0.1012),
(2, 5, -0.1105),
(2, 6, 0.1384),
(2, 7, -0.1572),
(2, 8, 0.1127),
(2, 9, -0.0808),
(2, 10, 0.1329),
(2, 11, -0.0989),
(2, 12, -0.1001),
(2, 13, 0.1106),
(2, 14, -0.0994),
(2, 15, 0.1919),
(2, 16, -0.1878),
(2, 17, 0.2727),
(2, 18, -0.2747),
(2, 19, 0.2394),
(2, 20, -0.2152),
(2, 21, 0.2088),
(3, 0, 0.0403),
(3, 1, -0.0718),
(3, 2, 0.1246),
(3, 3, -7.3723),
(3, 4, 0.1485),
(3, 5, 0.1020),
(3, 6, -0.1156),
(3, 7, 0.1531),
(3, 8, -0.1481),
(3, 9, 0.0612),
(3, 10, -0.1085),
(3, 11, 0.1428),
(3, 12, 0.0896),
(3, 13, -0.1212),
(3, 14, 0.0994),
(3, 15, -0.1757),
(3, 16, 0.1938),
(3, 17, -0.1840),
(3, 18, 0.2436),
(3, 19, -0.2839),
(3, 20, 0.2699),
(3, 21, -0.2396),
(4, 0, -0.0198),
(4, 1, 0.0743),
(4, 2, -0.1012),
(4, 3, 0.1485),
(4, 4, -7.4136),
(4, 5, -0.0985),
(4, 6, 0.1104),
(4, 7, -0.1277),
(4, 8, 0.1822),
(4, 9, -0.0540),
(4, 10, 0.0888),
(4, 11, -0.1698),
(4, 12, -0.0843),
(4, 13, 0.1265),
(4, 14, -0.0994),
(4, 15, 0.1713),
(4, 16, -0.1794),
(4, 17, 0.1982),
(4, 18, -0.2015),
(4, 19, 0.2295),
(4, 20, -0.2834),
(4, 21, 0.3272),
(5, 0, -0.2037),
(5, 1, 0.1338),
(5, 2, -0.1105),
(5, 3, 0.1020),
(5, 4, -0.0985),
(5, 5, -5.7633),
(5, 6, 0.1575),
(5, 7, -0.0966),
(5, 8, 0.0741),
(5, 9, -0.3941),
(5, 10, 0.2050),
(5, 11, -0.2026),
(5, 12, -0.1777),
(5, 13, 0.0826),
(5, 14, -0.1650),
(5, 15, 0.3434),
(5, 16, -0.2031),
(5, 17, 0.1309),
(5, 18, -0.0941),
(5, 19, 0.0748),
(5, 20, -0.0645),
(5, 21, 0.0599),
(6, 0, 0.1139),
(6, 1, -0.1703),
(6, 2, 0.1384),
(6, 3, -0.1156),
(6, 4, 0.1104),
(6, 5, 0.1575),
(6, 6, -5.6589),
(6, 7, 0.1527),
(6, 8, -0.1224),
(6, 9, 0.2590),
(6, 10, -0.3373),
(6, 11, 0.2053),
(6, 12, 0.1370),
(6, 13, -0.1234),
(6, 14, 0.1650),
(6, 15, -0.0828),
(6, 16, 0.2162),
(6, 17, -0.2049),
(6, 18, 0.1574),
(6, 19, -0.1189),
(6, 20, 0.0988),
(6, 21, -0.0918),
(7, 0, -0.1037),
(7, 1, 0.1068),
(7, 2, -0.1572),
(7, 3, 0.1531),
(7, 4, -0.1277),
(7, 5, -0.0966),
(7, 6, 0.1527),
(7, 7, -5.6537),
(7, 8, 0.1885),
(7, 9, -0.2128),
(7, 10, 0.2939),
(7, 11, -0.2949),
(7, 12, -0.1098),
(7, 13, 0.1505),
(7, 14, -0.1650),
(7, 15, 0.0753),
(7, 16, -0.0567),
(7, 17, 0.1401),
(7, 18, -0.1965),
(7, 19, 0.1976),
(7, 20, -0.1662),
(7, 21, 0.1383),
(8, 0, 0.0975),
(8, 1, -0.1079),
(8, 2, 0.1127),
(8, 3, -0.1481),
(8, 4, 0.1822),
(8, 5, 0.0741),
(8, 6, -0.1224),
(8, 7, 0.1885),
(8, 8, -5.7066),
(8, 9, 0.2029),
(8, 10, -0.2327),
(8, 11, 0.3660),
(8, 12, 0.0962),
(8, 13, -0.1641),
(8, 14, 0.1650),
(8, 15, -0.0532),
(8, 16, 0.0787),
(8, 17, -0.0788),
(8, 18, 0.1067),
(8, 19, -0.1634),
(8, 20, 0.2252),
(8, 21, -0.2647),
(9, 0, -0.2022),
(9, 1, 0.1227),
(9, 2, -0.0808),
(9, 3, 0.0612),
(9, 4, -0.0540),
(9, 5, -0.3941),
(9, 6, 0.2590),
(9, 7, -0.2128),
(9, 8, 0.2029),
(9, 9, -1.7615),
(9, 10, 0.1060),
(9, 11, -0.0751),
(9, 12, -0.1884),
(9, 13, 0.1120),
(9, 14, -0.1069),
(9, 15, 0.2330),
(9, 16, -0.1857),
(9, 17, 0.1550),
(9, 18, -0.1361),
(9, 19, 0.1253),
(9, 20, -0.1197),
(9, 21, 0.1173),
(10, 0, 0.0591),
(10, 1, -0.1315),
(10, 2, 0.1329),
(10, 3, -0.1085),
(10, 4, 0.0888),
(10, 5, 0.2050),
(10, 6, -0.3373),
(10, 7, 0.2939),
(10, 8, -0.2327),
(10, 9, 0.1060),
(10, 10, -1.7121),
(10, 11, 0.1245),
(10, 12, 0.1426),
(10, 13, -0.1578),
(10, 14, 0.1069),
(10, 15, -0.1100),
(10, 16, 0.1613),
(10, 17, -0.1779),
(10, 18, 0.1741),
(10, 19, -0.1612),
(10, 20, 0.1479),
(10, 21, -0.1398),
(11, 0, -0.0512),
(11, 1, 0.0583),
(11, 2, -0.0989),
(11, 3, 0.1428),
(11, 4, -0.1698),
(11, 5, -0.2026),
(11, 6, 0.2053),
(11, 7, -0.2949),
(11, 8, 0.3660),
(11, 9, -0.0751),
(11, 10, 0.1245),
(11, 11, -1.7430),
(11, 12, -0.1197),
(11, 13, 0.1807),
(11, 14, -0.1069),
(11, 15, 0.1164),
(11, 16, -0.1125),
(11, 17, 0.1266),
(11, 18, -0.1493),
(11, 19, 0.1730),
(11, 20, -0.1920),
(11, 21, 0.2024),
(12, 0, -0.1370),
(12, 1, 0.1159),
(12, 2, -0.1001),
(12, 3, 0.0896),
(12, 4, -0.0843),
(12, 5, -0.1777),
(12, 6, 0.1370),
(12, 7, -0.1098),
(12, 8, 0.0962),
(12, 9, -0.1884),
(12, 10, 0.1426),
(12, 11, -0.1197),
(12, 12, -1.3403),
(12, 13, 0.1553),
(12, 14, -0.3841),
(12, 15, 0.1808),
(12, 16, -0.1518),
(12, 17, 0.1276),
(12, 18, -0.1082),
(12, 19, 0.0937),
(12, 20, -0.0840),
(12, 21, 0.0792),
(13, 0, 0.0737),
(13, 1, -0.0948),
(13, 2, 0.1106),
(13, 3, -0.1212),
(13, 4, 0.1265),
(13, 5, 0.0826),
(13, 6, -0.1234),
(13, 7, 0.1505),
(13, 8, -0.1641),
(13, 9, 0.1120),
(13, 10, -0.1578),
(13, 11, 0.1807),
(13, 12, 0.1553),
(13, 13, -1.3403),
(13, 14, 0.3841),
(13, 15, -0.0550),
(13, 16, 0.0840),
(13, 17, -0.1082),
(13, 18, 0.1276),
(13, 19, -0.1421),
(13, 20, 0.1518),
(13, 21, -0.1566),
(14, 0, -0.0994),
(14, 1, 0.0994),
(14, 2, -0.0994),
(14, 3, 0.0994),
(14, 4, -0.0994),
(14, 5, -0.1650),
(14, 6, 0.1650),
(14, 7, -0.1650),
(14, 8, 0.1650),
(14, 9, -0.1069),
(14, 10, 0.1069),
(14, 11, -0.1069),
(14, 12, -0.3841),
(14, 13, 0.3841),
(14, 14, -0.0784),
(14, 15, 0.1256),
(14, 16, -0.1256),
(14, 17, 0.1256),
(14, 18, -0.1256),
(14, 19, 0.1256),
(14, 20, -0.1256),
(14, 21, 0.1256),
(15, 0, 0.3873),
(15, 1, -0.2098),
(15, 2, 0.1919),
(15, 3, -0.1757),
(15, 4, 0.1713),
(15, 5, 0.3434),
(15, 6, -0.0828),
(15, 7, 0.0753),
(15, 8, -0.0532),
(15, 9, 0.2330),
(15, 10, -0.1100),
(15, 11, 0.1164),
(15, 12, 0.1808),
(15, 13, -0.0550),
(15, 14, 0.1256),
(15, 15, -4.4140),
(15, 16, 0.1823),
(15, 17, -0.1061),
(15, 18, 0.0598),
(15, 19, -0.0311),
(15, 20, 0.0155),
(15, 21, -0.0088),
(16, 0, -0.2645),
(16, 1, 0.3106),
(16, 2, -0.1878),
(16, 3, 0.1938),
(16, 4, -0.1794),
(16, 5, -0.2031),
(16, 6, 0.2162),
(16, 7, -0.0567),
(16, 8, 0.0787),
(16, 9, -0.1857),
(16, 10, 0.1613),
(16, 11, -0.1125),
(16, 12, -0.1518),
(16, 13, 0.0840),
(16, 14, -0.1256),
(16, 15, 0.1823),
(16, 16, -4.2734),
(16, 17, 0.1291),
(16, 18, -0.0892),
(16, 19, 0.0654),
(16, 20, -0.0449),
(16, 21, 0.0334),
(17, 0, 0.2152),
(17, 1, -0.2660),
(17, 2, 0.2727),
(17, 3, -0.1840),
(17, 4, 0.1982),
(17, 5, 0.1309),
(17, 6, -0.2049),
(17, 7, 0.1401),
(17, 8, -0.0788),
(17, 9, 0.1550),
(17, 10, -0.1779),
(17, 11, 0.1266),
(17, 12, 0.1276),
(17, 13, -0.1082),
(17, 14, 0.1256),
(17, 15, -0.1061),
(17, 16, 0.1291),
(17, 17, -4.2310),
(17, 18, 0.1202),
(17, 19, -0.0853),
(17, 20, 0.0763),
(17, 21, -0.0696),
(18, 0, -0.1933),
(18, 1, 0.2230),
(18, 2, -0.2747),
(18, 3, 0.2436),
(18, 4, -0.2015),
(18, 5, -0.0941),
(18, 6, 0.1574),
(18, 7, -0.1965),
(18, 8, 0.1067),
(18, 9, -0.1361),
(18, 10, 0.1741),
(18, 11, -0.1493),
(18, 12, -0.1082),
(18, 13, 0.1276),
(18, 14, -0.1256),
(18, 15, 0.0598),
(18, 16, -0.0892),
(18, 17, 0.1202),
(18, 18, -4.2112),
(18, 19, 0.1298),
(18, 20, -0.1017),
(18, 21, 0.1057),
(19, 0, 0.1818),
(19, 1, -0.2015),
(19, 2, 0.2394),
(19, 3, -0.2839),
(19, 4, 0.2295),
(19, 5, 0.0748),
(19, 6, -0.1189),
(19, 7, 0.1976),
(19, 8, -0.1634),
(19, 9, 0.1253),
(19, 10, -0.1612),
(19, 11, 0.1730),
(19, 12, 0.0937),
(19, 13, -0.1421),
(19, 14, 0.1256),
(19, 15, -0.0311),
(19, 16, 0.0654),
(19, 17, -0.0853),
(19, 18, 0.1298),
(19, 19, -4.2163),
(19, 20, 0.1591),
(19, 21, -0.1307),
(20, 0, -0.1756),
(20, 1, 0.1919),
(20, 2, -0.2152),
(20, 3, 0.2699),
(20, 4, -0.2834),
(20, 5, -0.0645),
(20, 6, 0.0988),
(20, 7, -0.1662),
(20, 8, 0.2252),
(20, 9, -0.1197),
(20, 10, 0.1479),
(20, 11, -0.1920),
(20, 12, -0.0840),
(20, 13, 0.1518),
(20, 14, -0.1256),
(20, 15, 0.0155),
(20, 16, -0.0449),
(20, 17, 0.0763),
(20, 18, -0.1017),
(20, 19, 0.1591),
(20, 20, -4.2370),
(20, 21, 0.1832),
(21, 0, 0.1728),
(21, 1, -0.1877),
(21, 2, 0.2088),
(21, 3, -0.2396),
(21, 4, 0.3272),
(21, 5, 0.0599),
(21, 6, -0.0918),
(21, 7, 0.1383),
(21, 8, -0.2647),
(21, 9, 0.1173),
(21, 10, -0.1398),
(21, 11, 0.2024),
(21, 12, 0.0792),
(21, 13, -0.1566),
(21, 14, 0.1256),
(21, 15, -0.0088),
(21, 16, 0.0334),
(21, 17, -0.0696),
(21, 18, 0.1057),
(21, 19, -0.1307),
(21, 20, 0.1832),
(21, 21, -4.2862),
]

from qiskit.quantum_info import SparsePauliOp

num_qubits = 22

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
if abs(identity_coeff) > 1e-12:  
    paulis.append('I' * num_qubits)
    coeffs.append(identity_coeff)

# Build SparsePauliOp
H_JW = SparsePauliOp.from_list(list(zip(paulis, coeffs)))

# Print Hamiltonian
print(H_JW)
print('')
print('Number of Pauli terms =', len(H_JW))



# In[ ]:

#QiskitRuntimeService(channel = 'local')

from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit import QuantumCircuit, transpile
 
service = QiskitRuntimeService(channel = 'local')
backend = FakeFez()

ref_value = -8.762

params = np.array([5.09237035604333, -1.9816100626629274, 5.163170091034513, 4.207131156905029, 5.298418335905905, -1.245498635354769, -4.368336955629174, -4.345961598326135, -1.9646009930730488, -1.375809968825899, -1.769579411477366, -1.3679357745731857, 1.4102817362016278, 7.691328048740895, 4.861359559459467, 4.324789911488029, -5.13291328276058, 1.1070835894945406, -1.047249032392783, -2.1863320718937365, -5.497781332604764])

# Create quantum circuit
qc = QuantumCircuit(22)

# Initial gates
qc.x(0)
qc.h(0)
qc.cx(0, 1)
qc.ry(-params[0], 0)
qc.ry(-params[0], 1)
qc.cx(0, 1)
qc.h(0)

# Loop over qubits 2, 24
for i in range(2, 22):
    qc.h(i - 1)
    qc.cx(i - 1, i)
    qc.ry(-params[i - 1], i - 1)
    qc.ry(-params[i - 1], i)
    qc.cx(i - 1, i)
    qc.h(i - 1)

pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
isa_psi = pm.run(qc)
isa_observables = H_JW.apply_layout(isa_psi.layout)

print("No. of qubits:", isa_psi.num_qubits)
print("No. of gates:", isa_psi.count_ops())


estimator = Estimator(mode=backend)
 
# --- Estimator Loop ---
runs = []
eigenvalues = []
for run in range(10):
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
for run in range(10):
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
for run in range(10):
    job = estimator.run([(modified_circuit, isa_observables)])
    result = job.result()
    pub_result = result[0]
    eigenvalue = float(pub_result.data.evs)
    relative_error = abs((eigenvalue - ref_value) / ref_value)*100

    runs.append(run)
    eigenvalues.append(eigenvalue)

    # Print with comma-separated values: run, eigenvalue, relative_error
    print(f"({run},{eigenvalue},{relative_error}),")       
        
