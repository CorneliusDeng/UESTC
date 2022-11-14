"""
Tensors are a specialized data structure that are very similar to arrays and matrices.
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators.
In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data.
Tensors are also optimized for automatic differentiation.
"""

import torch
import numpy as np

"""
Initializing a Tensor
Tensors can be initialized in various ways. Take a look at the following examples:
"""
# Tensors can be created directly from data. The data type is automatically inferred.
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
# Tensors can be created from NumPy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
