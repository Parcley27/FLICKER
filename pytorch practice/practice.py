# From https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

import torch
import numpy as np

# Tensor init
displaySection = False
if displaySection:
    print("* Tensor Init *")

    data = [[1, 2], [3, 4]]
    xData = torch.tensor(data)

    npArray = np.array(data)
    xNp = torch.from_numpy(npArray)

    print(xData)
    print(xNp)

    xOnes = torch.ones_like(xData) # retains the properties of xData

    print(f"Ones Tensor: \n {xOnes} \n")

    xRand = torch.rand_like(xData, dtype=torch.float) # overrides the datatype of xData

    print(f"Random Tensor: \n {xRand} \n")

    shape = (2, 3)
    randomTensor = torch.rand(shape)
    onesTensor = torch.ones(shape)
    zerosTensor = torch.zeros(shape)

    print(f"Random Tensor: \n {randomTensor} \n")
    print(f"Ones Tensor: \n {onesTensor} \n")
    print(f"Zeros Tensor: \n {zerosTensor}")

# Tensor attributes
displaySection = False
if displaySection:
    print("* Tensor Attributes *")

    tensor = torch.rand(3, 4)

    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor datatype: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

# Tensor operations
displaySection = False

if displaySection:
    print("* Tensor Operations *")

    tensor = torch.rand(3, 4)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    tensor = tensor.to(device)

    print(f"Device tensor is stored on: {tensor.device}")

    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0

    print(tensor)

    combinedTensor = torch.cat([tensor, tensor, tensor], dim = 1)

    print(combinedTensor)

    print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
    print(f"tensor * tensor \n {tensor * tensor}")

    print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
    print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

    print(tensor, "\n")

    print(tensor.add(5), "\n")
    print(tensor, "\n")

    tensor.add_(5)
    print(tensor, "\n")

# Tensors with Numpy
displaySection = True

if displaySection:
    print("* Tensors with Numpy *")

    n = np.ones(5)
    t = torch.from_numpy(n)

    np.add(n, 1, out = n)
    print(f"numpy: {n}")
    print(f"torch: {t}")
