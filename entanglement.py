from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)  # Entangle qubits
state = Statevector(qc)  # Get statevector without measurement
print(state)  # Expected: [0.707+0j, 0+0j, 0+0j, 0.707+0j] (|00⟩ + |11⟩)/√2
