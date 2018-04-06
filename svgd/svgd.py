import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

'''
Runs T number of iterations of Stein Variational
Descent to move the given initial parrticles in the
direction of the target desnity p. The step sizes
are determined by AdaGrad.

Input: p - target density
       k - kernel
       x - initial set of points
       T - number of iterations
       alpha - momentum constant
       fudge - AdaGrad fudge factor
       step - step size scale

Output: final set of points after T iterations.
'''
def svgd(p, k, x, T, alpha=0.9, fudge=1e-6, step=1e-1):
    assert len(x.shape) == 2 and x.shape[1] == 1
    n = x.shape[0]
    x = Variable(torch.Tensor(x), requires_grad = True)
    accumulated_grad = torch.zeros((n, 1))
    #import pdb; pdb.set_trace()

    for i in range(T):
        if i % 20 == 0:
            print(i)
            print(torch.mean(x).data)
            plt.hist(x.data.numpy(), bins=20)
            plt.show()
        grad_logp = grad_log(p, x)
        kernel, grad_kernel = k_matrix(k, x)
        phi = torch.matmul(kernel, grad_logp)/n + torch.mean(grad_kernel, dim=1).view(-1, 1)
        if i == 0:
            accumulated_grad += phi**2
        else:
            accumulated_grad = alpha * accumulated_grad + (1-alpha) * phi**2
        stepsize = fudge + torch.sqrt(accumulated_grad)
        newx = x.data + torch.div(step * phi, stepsize)
        x = Variable(newx, requires_grad=True)
    return x



'''
Returns RBF kernel.

Input: bandwith if the kernel - h
Output: a function that returns the RBF kernel
'''
def RBF(h):
    def kernel(x1,x2):
        return torch.exp(-(1/h) * torch.norm(x1 - x2))
    return kernel

'''
Returns gradient of the RBF kernel
'''
def RBF_grad(x1,x2):
    return (2) * (x1 - x2) * torch.exp(-torch.norm(x1 -x2)**2)

'''
Takes the gradient of the log of the unnormalized
probability distribution.

Input: p - unnormalized probability distribution
       x - set of current particules

Output: gradient of log of distribution p at point x
'''
def grad_log(p, x):
    grad_log = []
    for i in range(len(x)):
        logp = torch.log(p(x[i]))
        logp.backward()
        grad_log.append(x.grad.data[i])
    grad_log = np.array(grad_log).reshape((-1, 1))
    return torch.Tensor(grad_log)


'''
Build the kernel matrix.

Input: k - kernel function
       x - set of current particules

Output: matrix containing the kernel evaluated at each pair of points
'''
def k_matrix(k, x):
    #import pdb; pdb.set_trace()
    n = len(x)
    kernel_matrix, grad_kernel_matrix = [], []
    for i in range(n):
        kernel_matrix.append([])
        grad_kernel_matrix.append([])
        for j in range(n):
            kernel_val = k(x[i], x[j])
            kernel_matrix[-1].append(float(kernel_val.data))
            #kernel_val.backward()
            kernel_grad = RBF_grad(x[i], x[j])
            grad_kernel_matrix[-1].append(float(kernel_grad.data))
            #x.grad.data.zero_()
    return torch.Tensor(kernel_matrix), torch.Tensor(grad_kernel_matrix)
