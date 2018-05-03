import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

# ## debugging
def numpy_p(x):
    return 1/3 * norm.pdf(x, -2, 1) + 2/3 * norm.pdf(x, 2, 1)

def numpy_p_far(x):
    return 1/2 * norm.pdf(x, -10, 1) + 1/2 * norm.pdf(x, 10, 1)

def numpy_p_complex(x):
    return (1/7) * norm.pdf(x, -2,3) + (2/7)  * norm.pdf(x, 2,1) + (3/7)  * norm.pdf(x, 5,5) + (1/7)  * norm.pdf(x, 6,0.5)

'''
Runs T number of iterations of Stein Variational
Descent to move the given initial parrticles in thegr
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
def svgd(p, kern, x, T, alpha=0.9, fudge=1e-6, step=1e-1):
    assert len(x.shape) == 2
    n, d = x.shape
    x = torch.Tensor(x)
    ## Put the most likely x at the front of the array (leading gradient).
    x = put_max_first(x, p)
    accumulated_grad = torch.zeros((n, d))
    for i in range(T):
        ### debugging
        print(i)
        if i % 50 == 0 or i == T-1:
            plt.figure()
            xs = np.arange(-20, 20, 0.01)
            plt.plot(xs, numpy_p_complex(xs), 'r-')
            g = gaussian_kde(x.numpy().reshape(-1))
            plt.plot(xs, g(xs), 'g')
            plt.savefig('../../../nparticles/{}.png'.format(i))
        varx = Variable(x, requires_grad = True)
        grad_logp = grad_log(p, varx)
        kernel, grad_kernel = kern(x)
        phi = torch.matmul(kernel, grad_logp)/n + torch.mean(grad_kernel, dim=1).view(n, d)
        # print(grad_logp)
        if i == 0:
            accumulated_grad += phi**2
        else:
            accumulated_grad = alpha * accumulated_grad + (1-alpha) * phi**2
        stepsize = fudge + torch.sqrt(accumulated_grad)
        x = x + torch.div(step * phi, stepsize)
        # break
    return x

'''
Build the RBF kernel matrix and grad_kernel matrix
given a set of points x.

Input: n x d set of points - x
Output: n x n kernel matrix, n x n x d grad_kernel matrix
'''
def RBF_kernel(x):
    n, d = tuple(x.size())
    pairdist = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pairdist[i][j] = torch.dist(x[i], x[j])
    med = torch.median(pairdist)
    h = med**2 / np.log(n)

    kernel = torch.exp(-(1/h) * pairdist**2)
    grad_kernel = torch.zeros((n, n, d))
    for i in range(n):
        for j in range(n):
            grad_kernel[i][j] = 2/h * kernel[i][j] * (x[i]-x[j])
    return torch.Tensor(kernel), torch.Tensor(grad_kernel)

'''
Returns RBF kernel.

Input: bandwith of the kernel - h
Output: a function that returns the RBF kernel
'''
def RBF(h):
    def kernel(x1,x2):
        return np.exp(-(1/h) * torch.norm(x1 - x2)**2)
    return kernel

'''
Returns the gradient of the RBF kernel w.r.t. x2
'''
def RBF_grad(x1,x2):
    return (2) * (x1 - x2) * np.exp(-torch.norm(x1 -x2)**2)

'''
Takes the gradient of the log of the unnormalized
probability distribution.

Input: p - unnormalized probability distribution
       x - set of current particles

Output: gradient of log of distribution p at point x
'''
def grad_log(p, x):
    n, d = tuple(x.size())
    grad_log = []
    for i in range(len(x)):
        if float(p(x[i])) < 1e-35:
            grad_log.append(grad_log[-1])
            continue
        # print ("p(x) is {}".format(p(x[i])))
        logp = torch.log(p(x[i]))
        # print ("logp us {}".format(logp))
        logp.backward()
        # print ("the grad is {}".format(x.grad.data[i]))
        grad_log.append(x.grad.data[i])
    grad_log = np.array(grad_log).reshape((n, d))
    return torch.Tensor(grad_log)


'''
Build the kernel matrix.

Input: k - kernel function
       x - set of current particules

Output: matrix containing the kernel evaluated at each pair of points
'''
def k_matrix(k, k_grad, x):
    #import pdb; pdb.set_trace()
    n = len(x)
    kernel_matrix, grad_kernel_matrix = [], []
    for i in range(n):
        kernel_matrix.append([])
        grad_kernel_matrix.append([])
        for j in range(n):
            kernel_val = k(x[i], x[j])
            kernel_matrix[-1].append(float(kernel_val))
            #kernel_val.backward()
            kernel_grad = k_grad(x[i], x[j])
            grad_kernel_matrix[-1].append(float(kernel_grad))
            #x.grad.data.zero_()
    return torch.Tensor(kernel_matrix), torch.Tensor(grad_kernel_matrix)

'''
Returns the same array with the element with highest probability first.

Input: x - array of samples
       p - probability distribution
Ourput: arrays of samples with highest probability element first.
'''
def put_max_first(x, p):
    n = len(x)
    best_index = -1
    best_prob = -1
    for i in range(n):
        prob_i = p(x[i])
        if float(prob_i) > float(best_prob):
            best_prob = prob_i
            best_index = i
    tmp = x[0]
    x[0] = x[best_index]
    x[best_index] = tmp
    return x
