import torch
import numpy
from sklearn import datasets
import matplotlib.pyplot as plt

numpy.set_printoptions(threshold=numpy.nan)


cov = torch.Tensor(datasets.make_spd_matrix(100))
while cov[0][0] < 2 or cov[0][1] < 2:
    cov = torch.Tensor(datasets.make_spd_matrix(100))

# cov = torch.Tensor(datasets.make_spd_matrix(100))
print (cov.numpy())

plt.imshow(cov)#, interpolation=None)
plt.show()

cov = torch.Tensor(cov)
torch.save(cov, 'covariance_matrix.pt')
