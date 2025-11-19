# grover.py
# Grover's Algorithm for Qiskit 1.4.4 (No Plotting)
# Run with: python grover.py [--state <00|01|10|11>] [--nqubits <2|3>]
# Example: python grover.py --state 00 --nqubits 3

# Install: pip install qiskit==1.4.4 qiskit-aer

import argparse
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCPhaseGate
import numpy as np
import sys

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Grover's Algorithm")
    parser.add_argument('--state', default='00', 
                        help='Marked state (e.g., 11 for 2 qubits, 111 for 3 qubits)')
    parser.add_argument('--nqubits', type=int, default=2, choices=[2, 3],
                        help='Number of index qubits (2 or 3, default: 2)')
    return parser.parse_args()

def create_oracle(n_qubits, marked_state):
    """Create oracle circuit to mark the specified state"""
    try:
        oracle = QuantumCircuit(n_qubits + 1)
        # Flip qubits where marked_state has 0s
        for i, bit in enumerate(marked_state):
            if bit == '0':
                oracle.x(i)
        # Multi-controlled Z gate
        if n_qubits == 2:
            oracle.cz(0, 1)
        else:  # n_qubits == 3
            oracle.mcp(np.pi, [0, 1], 2)
        # Flip back
        for i, bit in enumerate(marked_state):
            if bit == '0':
                oracle.x(i)
        return oracle
    except Exception as e:
        print(f"Error creating oracle: {e}")
        sys.exit(1)

def create_diffuser(n_qubits):
    """Create diffusion operator"""
    try:
        diffuser = QuantumCircuit(n_qubits + 1)
        for q in range(n_qubits):
            diffuser.h(q)
        for q in range(n_qubits):
            diffuser.x(q)
        if n_qubits == 2:
            diffuser.cz(0, 1)
        else:  # n_qubits == 3
            diffuser.mcp(np.pi, [0, 1], 2)
        for q in range(n_qubits):
            diffuser.x(q)
        for q in range(n_qubits):
            diffuser.h(q)
        return diffuser
    except Exception as e:
        print(f"Error creating diffuser: {e}")
        sys.exit(1)

def create_grover_circuit(n_qubits=2, marked_state='11'):
    """Create Grover circuit for given N and marked state"""
    try:
        N = 2 ** n_qubits
        iterations = int(np.pi / 4 * np.sqrt(N))
        print(f"Building circuit: {n_qubits} index qubits, N={N}, marking |{marked_state}>, {iterations} iteration(s)")

        qc = QuantumCircuit(n_qubits + 1, n_qubits)

        # 1. Superposition
        print("  Applying Hadamard gates for superposition...")
        for q in range(n_qubits):
            qc.h(q)
        qc.x(n_qubits)
        qc.h(n_qubits)  # Workspace qubit to |->

        # 2. Grover iterations
        oracle = create_oracle(n_qubits, marked_state)
        diffuser = create_diffuser(n_qubits)
        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}:")
            print("    Applying oracle...")
            qc.compose(oracle, inplace=True)
            print("    Applying diffuser...")
            qc.compose(diffuser, inplace=True)

        # 3. Measure
        print("  Adding measurements...")
        qc.measure(range(n_qubits), range(n_qubits))

        return qc
    except Exception as e:
        print(f"Error building circuit: {e}")
        sys.exit(1)

def run_simulation(qc, shots=1024):
    """Run simulation and return counts"""
    try:
        print(f"\nSimulating with {shots} shots...")
        simulator = AerSimulator()
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        return counts
    except Exception as e:
        print(f"Simulation error: {e}")
        sys.exit(1)

def main():
    print("GROVER'S ALGORITHM (Qiskit 1.4.4)")
    print("=" * 40)

    # Parse arguments
    args = parse_args()
    n_qubits = args.nqubits
    marked_state = args.state

    # Validate marked state length
    if len(marked_state) != n_qubits:
        print(f"Error: Marked state must be {n_qubits} bits (got {marked_state})")
        sys.exit(1)

    # Build circuit
    qc = create_grover_circuit(n_qubits=n_qubits, marked_state=marked_state)

    # Print circuit
    print("\nCircuit Diagram:")
    print(qc.draw('text', fold=-1))

    # Simulate
    counts = run_simulation(qc, shots=1024)

    # Print results
    print("\nResults:")
    total_shots = sum(counts.values())
    for state, count in sorted(counts.items()):
        prob = count / total_shots * 100
        print(f"  |{state}> : {count}/{total_shots} ({prob:.1f}%)")

    # Success check
    success_prob = counts.get(marked_state, 0) / total_shots * 100
    print(f"\nResult: |{marked_state}> found with {success_prob:.1f}% probability")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Main error: {e}")
        sys.exit(1)
