import numpy as np
import numpy.matlib as nm
import svgd
import torch
from torch.distributions import Normal
#import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm


def OneDimNormalMixture(x):
    return (1/3) * torch.exp(Normal(-2,1).log_prob(x)) + (2/3) * torch.exp(Normal(2,1).log_prob(x))

def SimpleTwoDim(x):
    mean = torch.Tensor([0, 0])
    covariance = torch.Tensor([[1, -10], [-10, 1]])
    #return torch.exp()


x = np.random.normal(-10, 1, 200).reshape((-1, 2))

result = svgd.svgd(OneDimNormalMixture, svgd.RBF_kernel, x, 500)
# g = gaussian_kde(result.numpy().reshape(-1))
# xs = np.arange(-10, 10, 0.01)
# plt.plot(xs, svgd.numpy_p(xs), 'r-')
# plt.plot(xs, g(xs), 'g')
# plt.show()
#plt.hist(result.data.numpy(), bins=20)
#plt.show()
