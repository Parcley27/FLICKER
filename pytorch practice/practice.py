# From https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

import torch
import numpy as np

# Tensor init
displaySection = False

if displaySection:
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
displaySection = True

if displaySection: 
    tensor = torch.rand(3, 4)

    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor datatype: {tensor.dtype}")
    print(f"Device tensor is stored in: {tensor.device}")


