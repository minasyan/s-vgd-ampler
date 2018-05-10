import numpy as np
import numpy.matlib as nm
import svgd
import torch
import time
from sklearn import datasets
from torch.distributions import Normal
from torch.distributions import multivariate_normal
#import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm


def OneDimNormalMixture(x):
    return (1/3) * torch.exp(Normal(-2,1).log_prob(x)) + (2/3) * torch.exp(Normal(2,1).log_prob(x))

def OneDimNormalMixtureFar(x):
    return (1/2) * torch.exp(Normal(-10,1).log_prob(x)) + (1/2) * torch.exp(Normal(10,1).log_prob(x))

def OneDimNormalMixtureComplex(x):
    return (1/7) * torch.exp(Normal(-2,3).log_prob(x)) + (2/7)  * torch.exp(Normal(2,1).log_prob(x)) + (3/7)  * torch.exp(Normal(5,5).log_prob(x)) + (1/7)  * torch.exp(Normal(6,0.5).log_prob(x))

def SimpleTwoDim(x):
    mean = torch.Tensor([5, 5])
    # covariance = torch.Tensor([[1,10],[10,1]])
    covariance = torch.tensor([[ 1.7313,  1.0120], [ 1.0120,  1.7964]])
    gauss = multivariate_normal.MultivariateNormal(mean, covariance)
    return torch.exp(gauss.log_prob(x))
    #return torch.exp()


def generate100Dim(mean, covariance):
    def Complex100Dim(x):
        gauss = multivariate_normal.MultivariateNormal(mean, covariance)
        return gauss.log_prob(x)
    return Complex100Dim

# print ("The variance is {}".format(np.var(result.numpy().reshape(-1))))
# usable_res = result.numpy().reshape(-1)
# left_res = []
# right_res = []
# for val in usable_res:
#     if val < 0:
#         left_res.append(val)
#     else:
#         right_res.append(val)
# print("The Variances are {} for left and {} for right".format(np.var(left_res), np.var(right_res)))


# g = gaussian_kde(result.numpy().reshape(-1))
# xs = np.arange(-10, 10, 0.01)
# plt.plot(xs, svgd.numpy_p(xs), 'r-')
# plt.plot(xs, g(xs), 'g')
# plt.show()
#plt.hist(result.data.numpy(), bins=20)
#plt.show()

if __name__=='__main__':
    dimension = 1
    nparticles = 200
    x = np.random.normal(0, 1, nparticles * dimension).reshape((-1, dimension))

    ## TODO: only for 100DIM example
    mean = torch.Tensor([2.5]*dimension)
    covariance = torch.Tensor(datasets.make_spd_matrix(dimension))

    start_time = time.time()
    result = svgd.svgd(generate100Dim(mean, covariance), svgd.RBF_kernel, x, 100, mean=mean, covariance=covariance)
    print("--- %s seconds ---" % (time.time() - start_time))
