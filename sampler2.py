from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Create the circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Get the statevector
state = Statevector.from_instruction(qc)

# Get probabilities of measuring each basis state
probs = state.probabilities_dict()

print(probs)
