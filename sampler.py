from qiskit import QuantumCircuit
from qiskit.primitives import Sampler

# Create a simple quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Initialize the Sampler
sampler = Sampler()

# Run the circuit
result = sampler.run(qc).result()

# Get the quasi-probabilities
quasi_probs = result.quasi_dists[0]
print(quasi_probs)

