# import torch
# import numpy
# from sklearn import datasets
# from scipy.stats import wishart, invgamma
# import matplotlib.pyplot as plt
#
# numpy.set_printoptions(threshold=numpy.nan)
#
#
# d = 100

# cov = torch.Tensor(datasets.make_spd_matrix(100))
# while cov[0][0] < 2 or cov[0][1] < 2:
#     cov = torch.Tensor(datasets.make_spd_matrix(100))
#
# # cov = torch.Tensor(datasets.make_spd_matrix(100))
# print (cov.numpy())
#
# plt.imshow(cov)#, interpolation=None)
# plt.show()
#
# cov = torch.Tensor(cov)
# torch.save(cov, 'covariance_matrix.pt')


# identity = numpy.identity(d)
# x = numpy.linspace(1e-5, 8, 100)
# w = wishart.pdf(x, df=100, scale=identity)
# print(w)

import numpy as np
from random import choice
import matplotlib.pyplot as plt
import torch

d= 100
a = 2

A = np.matrix([np.random.randn(d) + np.random.randn(1) * a for i in range(d)])
A = A * np.transpose(A)
D_half = np.diag(np.diag(A)**(-0.5))
C = D_half*A*D_half

# vals = list(np.array(C.ravel())[0])
# plt.hist(vals, range=(-1,1))
# plt.show()
# plt.imshow(C, interpolation=None)
# plt.show()

print(C.shape)
print(C)

variances = np.arange(0.01, 1.01, 0.01)
std = np.diag(np.sqrt(variances))


cov = np.dot(np.dot(std, C), std)
vals = list(np.array(cov.ravel())[0])
plt.hist(vals, range=(-1,1))
plt.show()
plt.imshow(cov, interpolation=None)
plt.show()

cov = torch.Tensor(cov)
torch.save(cov, 'final_covariance_matrix.pt')
