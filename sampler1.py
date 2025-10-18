from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

# Create a quantum circuit with classical registers
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all(inplace=True)  # Required for StatevectorSampler to produce output

# Initialize the sampler
sampler = StatevectorSampler()

# Run the sampler
job = sampler.run([qc])
result = job.result()

# Access the result for the first circuit (PUB)
pub_result = result[0]
#probs = result[0].quasi_probs

# The classical register is named 'meas' by default when using measure_all()
# Get the counts from the 'meas' register
#counts = pub_result.data.meas.get_counts()

# Convert integer keys to bitstrings

#num_qubits = qc.num_qubits
#bitstrings = dict(counts)
#print(bitstrings)

# Get counts from the default classical register 'meas'
counts = pub_result.data["meas"].get_counts()

print(counts)

