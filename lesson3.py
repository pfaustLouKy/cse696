import numpy as np
 
ket0 = np.array([[1], [0]])
ket1 = np.array([[0], [1]])
 
print(ket0 / 2 + ket1 / 2)

M1 = np.array([[1, 1], [0, 0]])
M2 = np.array([[1, 0], [0, 1]])
M = M1 / 2 + M2 / 2
print(M)

print(np.matmul(M1, ket1))
print(np.matmul(M1, M2))
print(np.matmul(M, M))

from qiskit.visualization import array_to_latex

display(array_to_latex(np.matmul(M1, ket1)))
display(array_to_latex(np.matmul(M1, M2)))
display(array_to_latex(np.matmul(M, M)))
