from qiskit import QuantumCircuit
qc = QuantumCircuit(1)
qc.x(0)  # X gate
qc.x(0).inverse()  # X† (adjoint, same as X)
