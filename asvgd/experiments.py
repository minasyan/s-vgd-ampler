import sys
sys.path.append('../')
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from svgd.experiments import OneDimNormalMixture
from svgd.svgd import RBF_kernel
from asvgd import asvgd



def nn(nin, nhidden1, nhidden2, nout, params):
	nweights1 = nin * nhidden1
	nweights2 = nhidden1 * nhidden2 + nweights1
	nweights3 = nhidden2 * nout + nweights2
	assert nweights3 == params.size()[0]
	# inputs is given as m x nin with m batch size.
	# uses sigmoid for now, can be changed to relu.
	def f(inputs, params):
		assert nin == inputs.size()[1]
		a = F.sigmoid(torch.matmul(inputs, params[:nweights1].view(nin, nhidden1)))
		b = F.sigmoid(torch.matmul(a, params[nweights1:nweights2].view(nhidden1, nhidden2)))
		c = torch.matmul(b, params[nweights2:nweights3].view(nhidden2, nout))
		return c
	return f

if __name__ == '__main__':
	nin, nhidden1, nhidden2, nout = 10, 20, 10, 1
	T = 100
	m = 100
	def q(m):
		dist = Normal(0, 1)
		return dist.sample(torch.Size([m, nin]))
	nparams = nin * nhidden1 + nhidden1 * nhidden2 + nhidden2 * nout
	params = Normal(0, 1).sample(torch.Size([nparams]))
	f = nn(nin, nhidden1, nhidden2, nout, params)
	result_params = asvgd(OneDimNormalMixture, f, q, RBF_kernel, params, T, m)
	result = f(q(m), result_params)
	print(torch.mean(result))
