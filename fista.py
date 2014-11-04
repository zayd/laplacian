""" Implementation of inference algorithms for sparse coding """

import numpy as np
import math
import scipy.sparse as sps
import scipy.sparse.linalg
import time

def fista(I, Phi, lambdav, max_iterations=150, display=False, problem='l1', groups=None):
  """
  FISTA Inference for Lasso (l1) Problem
  I: Batches of images (dim x batch)
  Phi: Dictionary (dim x dictionary element) (nparray or sparse array)
  lambdav: Sparsity penalty
  max_iterations: Maximum number of iterations
  """
  x = np.zeros((Phi.shape[1], I.shape[1]))
  Q = Phi.T.dot(Phi)
  c = -2*Phi.T.dot(I)

  L = scipy.sparse.linalg.eigsh(2*Q, 1, which='LM')[0]
  invL = 1/float(L)
  y = x
  t = 1

  for i in range(max_iterations):
    g = 2*Q.dot(y) + c

    if problem == 'l1':
      x2 = l1proxOp(y-invL*g,invL*lambdav)
    elif problem == 'l1l2':
      x2 = l1l2proxOp(y-invL*g, groups, invL*lambdav)

    t2 = (1+math.sqrt(1+4*(t**2)))/2.0
    y = x2 + ((t-1)/t2)*(x2-x)
    x = x2
    t = t2
    if display == True:
      print "L1 Objective " +  str(np.sum((I-Phi.dot(x2))**2) + lambdav*np.sum(np.abs(x2)))
      # (num_indices, num_groups) = groups.shape

      #penalty = 0
      #for i in range(num_groups):
      # penalty += np.linalg.norm(x2[groups[i]])

      #print "L1L2 Objective" + str(np.sum((I-Phi.dot(x2))**2) + lambdav*penalty)

  return x2

def l1proxOp(x,t):
  """ L1 Proximal Operator """
  return np.fmax(x-t, 0) + np.fmin(x+t, 0)

def l1l2proxOp(x, g, t):
  """ Proximal Operator of l1\l2 group sparse coding
    x: Coefficient vector
    g: Group indices. (groups x indices)
    t: penalty
  """
  (num_indices, num_groups) = g.shape

  for i in range(num_groups):
    group_norm = np.linalg.norm(x[g[i]])

    if group_norm <= t:
      x[g[i]] = 0
    else:
      x[g[i]] *= (group_norm-t)/float(group_norm)

  return x
