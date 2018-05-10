import sys
sys.path.append('../')
import torch
import numpy as np
from svgd.experiments import OneDimNormalMixture, OneDimNormalMixtureFar, OneDimNormalMixtureComplex
from torch.distributions.normal import Normal
from torch.autograd import Variable
from tqdm import tqdm

def grad(F, x):
	grads = torch.zeros(x.size())
	for i in range(x.size()[0]):
		xi = Variable(x[i], requires_grad=True)
		ans = F(xi)
		ans.backward()
		grads[i] = xi.grad.data
	return grads

'''
Hamiltonian dynamics transformation of a point
using leapfrog integration w/o M-H step.

Input:	U - the potential energy function
		L - the number of leapfrog steps
		eps - the step size
		current_q - the given point q

Output:	current_q - the same point q
		current_p - the momentum for the given point q
		q - the transformed point q
		p - the momentum for the transformed point q
'''
def hmc_transform(U, L, eps, current_q):
	current_p = Normal(0, 1).sample(current_q.size())
	p, q = current_p, current_q

	p = p - eps * grad(U, q) / 2.0
	for i in range(L):
		q = q + eps * p
		if i != L - 1:
			p = p - eps * grad(U, q)
	p = p - eps * grad(U, q) / 2.0

	return (current_q, current_p), (q, -p)

'''
A neural network imitating an HMC sampler.

Input:	layers - number of stacked hmc_transform layers
		p - target distribution, log of the pdf
		L - fixed number of steps for HMC
		params - stepsize parameters for the sampler network

Output: f - neural network imitating HMC w/ stepsize parameters
'''
def hmc_nn(layers, p, L, params):
	assert layers == params.size()[0]
	U = lambda x: -p(x)
	def f(inputs, params):
		layer_in = inputs
		for i in range(layers):
			old, new = hmc_transform(U, L, params[i], layer_in)
			layer_in = new[0]
		return layer_in
	return f

'''
HMC sampler.

Input:	p - target distribution, log of the pdf
		L - fixed number of steps
		eps - fixed stepsize
		nsamples - number of samples
		d - dimension of samples

Output: samples - list of the samples from HMC
		accepted - the acceptance rate during the HMC run
'''
def hmc_sampler(p, L, eps, nsamples, d):
	U = lambda x: -p(x)
	samples = []
	current = Normal(0, 1).sample(torch.Size([1, d]))
	accepted = 0
	for i in tqdm(range(nsamples)):
		old, new = hmc_transform(U, L, eps, current)
		old_U = U(old[0])
		new_U = U(new[0])
		old_K = torch.sum(old[1]**2)/2
		new_K = torch.sum(new[1]**2)/2
		prob = float(torch.exp(old_U - new_U + old_K - new_K))
		if np.random.rand() < prob:
			accepted += 1
			current = new[0]
		else:
			current = old[0]
		samples.append(current.numpy())
	accepted = accepted / nsamples
	return samples, accepted

'''
Choose the optimal value of stepsize according to the acceptance rate.
Best acceptance rate should be around 65% (cite).

Input:	p - target distribution, log of the pdf
		L - fixed number of steps
		eps - list of possible values for the stepsize
		nsamples - number of samples to generate
		d - dimension of samples

Output:	best_eps - optimal value of the stepsize
		best_rate - corresponding acceptance rate
'''
def choose_best(p, L, eps, nsamples, d):
	baseline = 0.65
	best_rate = 0
	best_eps = -1
	for eps_val in eps:
		_, accepted = hmc_sampler(p, L, eps_val, nsamples, d)
		print("stepsize: {} gives acceptance rate: {}".format(eps_val, accepted))
		if abs(accepted - baseline) < abs(best_rate - baseline):
			best_rate = accepted
			best_eps = eps_val
	return best_eps, best_rate

if __name__ == '__main__':
	L = 10
	eps = 0.05
	nsamples = 10000
	d = 1
	p = OneDimNormalMixture
	samples = hmc_sampler(p, L, eps, nsamples, d)
	samples = np.array(samples).reshape((nsamples, d))
	print(np.mean(samples, axis=0))
