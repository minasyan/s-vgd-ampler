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

variances = np.arange(0.01, 1.01, 0.01)
std = np.diag(np.sqrt(variances))


cov = np.dot(np.dot(std, C), std)
vals = list(np.array(cov.ravel())[0])
plt.hist(vals, range=(-1,1))
plt.show()
plt.imshow(cov, interpolation=None)
plt.show()

C = torch.Tensor(C)
torch.save(C, 'final_correlation_matrix.pt')
cov = torch.Tensor(cov)
torch.save(cov, 'final_covariance_matrix.pt')
