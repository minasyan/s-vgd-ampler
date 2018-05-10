import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.autograd import Variable

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
		p - target distribution
		L - fixed number of steps for HMC
		params - stepsize parameters for the sampler network

Output: f - neural network imitating HMC w/ stepsize parameters
'''
def hmc_nn(layers, p, L, params):
	assert layers == params.size()[0]
	U = lambda x: -torch.log(p(x))
	def f(inputs, params):
		layer_in = inputs
		for i in range(layers):
			old, new = hmc_transform(U, L, params[i], layer_in)
			layer_in = new[0]
		return layer_in
	return f
