import torch
from sklearn import datasets
import numpy
numpy.set_printoptions(threshold=numpy.nan)



# while cov[0][0] < 2 or cov[0][1] < 1.5:
#     cov = torch.Tensor(datasets.make_spd_matrix(2))

cov = torch.Tensor(datasets.make_spd_matrix(100))
print (cov.numpy())
