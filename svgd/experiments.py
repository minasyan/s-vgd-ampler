import numpy as np
import numpy.matlib as nm
import svgd
import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt


def OneDimNormal(x):
    return (1/3) * torch.exp(Normal(-2,1).log_prob(x)) + (2/3) * torch.exp(Normal(2,1).log_prob(x))


x = np.random.normal(-10, 1, 100).reshape((-1, 1))
h = 1

result = svgd.svgd(OneDimNormal, svgd.RBF(1), x, 1000)
plt.hist(result.data.numpy(), bins=20)
plt.show()
