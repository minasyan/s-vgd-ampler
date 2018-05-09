import sys
sys.path.append('../')
import torch
import numpy as np
from torch.autograd import Variable
from svgd.svgd import grad_log, put_max_first
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

'''
Amortized SVGD that performs T iterations of SVGD steps on a parameterized
function (neural network) to update the parameters.

Input: 	p - target density
		f - neural network
		q - initial sampling distribution
		kern - kernel function returning kernel and grad_kernel matrices
		params - the parameters of f to be updated given as a flat 1D tensor
		T - number of iterations
		m - batch size

Output: params - final values of parameters
'''
def asvgd(p, f, q, kern, params, T, m, alpha=0.9, step=1e-1):
	dparam = params.size()[0]
	accumulated_grad = torch.zeros(params.size())
	fudge = 1e-6

	for t in range(T):
		print("iteration: {}".format(t))
		inputs = q(m)	# m x p
		zs = f(inputs, params)	# m x d
		print("mean is: {}".format(torch.mean(zs)))
		d = zs.size()[1]
		if t % 50 == 0:
			g = gaussian_kde(zs.numpy().reshape(-1))
			xs = np.arange(-20, 20, 0.01)
			plt.plot(xs, g(xs), 'g')
			plt.show()
		## put the most likely input at the front to lead the direction
		zs = put_max_first(zs, p)
		varz = Variable(zs, requires_grad = True)
		grad_logp = grad_log(p, varz)	# m x d
		kernel, grad_kernel = kern(zs)	# (m x m), (m x m x d)
		phi = torch.matmul(kernel, grad_logp)/m + torch.mean(grad_kernel, dim=1).view(m, d)	# m x d
		grad_params = get_gradient(f, inputs, params).view(m, dparam, d)	# m x dparam x d
		update = torch.zeros(params.size())
		for i in range(m):
			update += torch.matmul(grad_params[i], phi[i])
		if t == 0:
			accumulated_grad += update**2
		else:
			accumulated_grad = alpha * accumulated_grad + (1 - alpha) * update**2
		stepsize = fudge + torch.sqrt(accumulated_grad)
		params += torch.div(step * update, stepsize)
	return params

'''
Get gradient w.r.t. params of function f with inputs and params

Input:	f - neural network
		inputs - the inputs to f as a batch (m x d)
		params - the parameters of f as a vector (dparam)

Output:	grads - the gradients w.r.t. params (m x d x dparam)
'''
def get_gradient(f, inputs, params):
	dparam = params.size()[0]
	var_params = Variable(params, requires_grad = True)
	f_value = f(inputs, var_params)
	m, d = f_value.size()[0], f_value.size()[1]
	grads = torch.zeros(m, d, dparam)
	for i in range(m):
		for j in range(d):
			f_value[i][j].backward(retain_graph=True)
			grads[i][j] = var_params.grad.data
			var_params.grad.zero_()
	return grads
