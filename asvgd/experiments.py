import sys
sys.path.append('../')
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from svgd.experiments import OneDimNormalMixture, OneDimNormalMixtureFar, OneDimNormalMixtureComplex
from svgd.svgd import RBF_kernel
from asvgd import asvgd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from hmc import hmc_nn


def OneDimNormalMixture2(x):
    return (1/3) * torch.exp(Normal(2,1).log_prob(x)) + (2/3) * torch.exp(Normal(6,1).log_prob(x))

def OneDimNormalMixture3(x):
    return (1/3) * torch.exp(Normal(0,1).log_prob(x)) + (2/3) * torch.exp(Normal(8,1).log_prob(x))

def OneDimNormalMixture4(x):
    return (4/5) * torch.exp(Normal(0,1).log_prob(x)) + (1/5) * torch.exp(Normal(8,1).log_prob(x))

def nn(nin, nhidden1, nhidden2, nout, params):
	nweights1 = nin * nhidden1
	nweights2 = nhidden1 * nhidden2 + nweights1
	nweights3 = nhidden2 * nout + nweights2
	assert nweights3 == params.size()[0]
	# inputs is given as m x nin with m batch size.
	# uses sigmoid for now, can be changed to relu.
	def f(inputs, params):
		assert nin == inputs.size()[1]
		a = F.relu(torch.matmul(inputs, params[:nweights1].view(nin, nhidden1)))
		b = F.relu(torch.matmul(a, params[nweights1:nweights2].view(nhidden1, nhidden2)))
		c = torch.matmul(b, params[nweights2:nweights3].view(nhidden2, nout))
		return c
	return f

if __name__ == '__main__':
	nin, nhidden1, nhidden2, nout = 1, 5, 5, 1
	T = 1000
	m = 100
	def q(m):
		dist = Normal(0, 1)
		return dist.sample(torch.Size([m, nin]))
	nparams = nin * nhidden1 + nhidden1 * nhidden2 + nhidden2 * nout
	# params = Normal(0, 1).sample(torch.Size([nparams]))
	params = torch.load('params_third.pt')
	f = nn(nin, nhidden1, nhidden2, nout, params)
	result_params = asvgd(OneDimNormalMixture4, f, q, RBF_kernel, params, T, m)
	result = f(q(m*m), result_params)
	g = gaussian_kde(result.numpy().reshape(-1))
	xs = np.arange(-20, 20, 0.01)
	plt.plot(xs, g(xs), 'g')
	plt.show()
	print(torch.mean(result))
