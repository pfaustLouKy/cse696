import numpy as np

A = np.array([[1, 2],[3, 4]])
B = np.array([[0, 5],[6, 7]])
tensor_product = np.kron(A,B)

print(tensor_product)
