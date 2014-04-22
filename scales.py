# Implements sparse coding using tiling on a multiple layers of laplacian pyramid
import numpy as np
from numpy import mean, std, ceil, mod, floor, dot, arange
import scipy.sparse as sps
import scipy.sparse.linalg
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from math import sqrt
import time

import skimage.transform as skt
import sklearn.preprocessing as skp

import preprocess
import laplacian_pyramid
import fista
#import gpu

def learn(G=None, Phi=None, base_image_dim=32*32, patch_dim=9*9, scales=2, overcomplete=1, iterations=4000, inf_iterations=150, batch=100, 
    alpha=[200, 400], beta=0.95, gamma=0.95, decrease_every=200, lambdav=0.05, plot=False, plot_every=50, save=False, label=''):

    patch_side = int(sqrt(patch_dim))
    base_image_side = int(sqrt(base_image_dim))
    pad = (patch_side-1)/2
    
    indices = [all_indices(base_image_side/2**(scales-s-1), patch_side, overcomplete=overcomplete) for s in range(scales)]

    if Phi == None:
      Phi = initialize_scales(G, base_image_side, patch_side, scales=scales)

    if G == None: 
      G = laplacian_pyramid.generative(base_image_side, patch_side, scales, base_mask_radius=(patch_side-1)/2)

    base_neurons = base_image_dim
    momentum = [np.zeros((patch_dim, base_neurons/4**(scales-s-1))) for s in range(scales)]
    M = sps.hstack([G[s].dot(Phi[s]) for s in range(scales)]).tocsr()
    max_eig = scipy.sparse.linalg.svds(M, 1, which='LM', return_singular_vectors=False)

    for t in range(iterations+1):
      I = preprocess.extract_images(images='vanhateran', num_images=batch, image_dim=base_image_dim, normalize=True, lowpass=True, pad=pad).T

      A = inference(I, G, Phi, base_image_dim, lambdav, algorithm='fista-gpu', max_iterations=inf_iterations, max_eig=max_eig)
      R = reconstruct(G, Phi, A)

      error = I - R

      old_obj = np.sum(error**2) + lambdav*np.sum(np.sum(np.abs(a)) for a in A)
      print "Old Objective: " + str(old_obj)

      for s in range(scales):
        neurons = base_neurons/4**((scales-s-1))
        image_side = base_image_side/2**((scales-s-1))
        error_s = G[s].T.dot(error)

        error_s = error_s.reshape(image_side+2*pad, image_side+2*pad, batch)
        error_s = patchify(error_s, (patch_side, patch_side))
        error_s = error_s.reshape(batch, neurons, patch_dim)

        dPhi = (error_s.transpose(1,2,0) * A[s][:, None, :]).sum(axis=2).T

        gamma = 1 - 3/float(t+5)
        momentum[s] = gamma*momentum[s] + alpha[s]/float(batch) * dPhi 

        Phi[s] = Phi[s].tolil()
        Phi[s][indices[s][0], indices[s][1]] += momentum[s]
        Phi[s] = Phi[s].tocsc()

      Phi = normalize(G, Phi)
      #for s in range(scales):
      #  skp.normalize(Phi[s], norm='l2', axis=0, copy=False)

      R = reconstruct(G, Phi, A)
      
      error = I - R
      new_obj = np.sum(error**2) + lambdav*np.sum(np.sum(np.abs(a)) for a in A)
      print "New Objective: " + str(new_obj)

      # Armajillo's Rule
      if new_obj > old_obj or t % decrease_every == 0:   
        alpha = [a * beta for a in alpha]
        print "Learning rate: " + str(alpha)

      if t % plot_every == 0:
          display_scales(t, G, Phi, save=save, patch_side=patch_side, label=label)
          # Eigenvalue doesn't change that often
          M = sps.hstack([G[s].dot(Phi[s]) for s in range(scales)]).tocsr()
          max_eig = scipy.sparse.linalg.svds(M, 1, which='LM', return_singular_vectors=False)
          print "Max eigenvalue" + str(max_eig)

def reconstruct(G, Phi, A):
    scales = len(Phi)

    R = Phi[-1].dot(A[-1]) # Base is just id. matrix

    for s in range(scales-1):
        R += G[s].dot(Phi[s].dot(A[s]))

    return R

def reconstruct2(Phi, A):
  def upsample_blur(Phi, A, upscale):
    scales = len(Phi)

  R = np.zeros()
  for s in range(scales-1):
    Phi[s].dot(A[s])


def inference(I, G, Phi, base_image_dim, lambdav, algorithm='fista', max_iterations=100, max_eig=None):
  scales = len(Phi)

  if algorithm == 'fista':
    M = sps.hstack([G[s].dot(Phi[s]) for s in range(scales)]).tocsr()
    L = fista.fista(I, M, lambdav, max_iterations=max_iterations, display=False)
    A = np.vsplit(L, np.cumsum([base_image_dim/4**(scales-s-1) for s in range(scales)]))
    A.pop(-1)

  elif algorithm == 'fista-gpu':
    M = sps.hstack([G[s].dot(Phi[s]) for s in range(scales)]).tocsr()
    L = gpu.fista(I, M, lambdav, max_iterations=max_iterations, display=True, verbose=False, L=None)
    A = np.vsplit(L, np.cumsum([base_image_dim/4**(scales-s-1) for s in range(scales)]))
    A.pop(-1)

  return A

def initialize_scales(G, base_image_side, patch_side, scales):
  def initialize(image_side, patch_side, overcomplete=1, convolutional=False):
    """ Initialize sparse Phi matrix with Gaussian random noise """ 

    pad = patch_side-1
    mask_radius = pad/2
    pad_image_side = image_side+pad
    neurons = image_side**2

    indices = all_indices(image_side, patch_side, overcomplete)

    if convolutional == False:
      dPhi = np.random.randn(patch_side**2, neurons*overcomplete)
    else:
      dPhi = np.random.randn(patch_side**2, overcomplete)
      dPhi = dPhi.repeat(neurons, axis=1)

    Phi = sps.lil_matrix((pad_image_side**2, neurons*overcomplete))
    Phi[indices[0], indices[1]] = dPhi
      
    Phi = sps.csc_matrix(Phi)
    #skp.normalize(Phi, norm='l2', axis=0, copy=False)
    return Phi

  Phi = []
  for s in range(scales):
    Phi.append(initialize(base_image_side/(2**(scales-s-1)), patch_side, overcomplete=1, convolutional=False))

  Phi = normalize(G, Phi)

  return Phi
  
def all_indices(image_side, patch_side, overcomplete=1):
    """ Returns list of indices for all neurons for advanced indexing """
    def indices(center, patch_side, image_side):
        """ Return indices to use for advanced indexing for single neuron. 
        Assumes image is padded """
        x0, y0 = center

        indices = np.array((), 'int')
        for i in range(patch_side):
            indices = np.append(indices, np.arange(patch_side)+x0+(y0+i)*image_side)

        return indices

    pad = (patch_side-1)

    all_indices = np.zeros((patch_side**2,image_side**2), 'int')
    for o in range(overcomplete):
      for y in range(image_side):
          for x in range(image_side):
              all_indices[:,x + y*image_side] = indices((x, y), patch_side, image_side+pad)

    return [np.tile(all_indices, overcomplete), np.arange(overcomplete*(image_side**2))]

# Helper Functions
def normalize(G, Phi):
  """ Normalize dictionary elements based on norm when projecting to image """
  scales = len(Phi)

  skp.normalize(Phi[-1], norm='l2', axis=0, copy=False)

  for s in range(scales-1):
    Q = G[s].dot(Phi[s]).tocsc()
    Q.data **= 2
    n = 1/np.sqrt(Q.sum(axis=0))
    (_, l) = n.shape
    # N = np.sqrt(np.sum(G[s].dot(Phi[s])**2, axis=0))
    N = sps.dia_matrix((n, 0), shape=(l, l))
    Phi[s] = Phi[s].dot(N)

  return Phi

def reshape(a, shape):
  """Reshape the sparse matrix a to shape """ 
  c = a.tocoo()
  nrows, ncols = c.shape
  size = nrows * ncols

  new_size =  shape[0] * shape[1]
  if new_size != size:
      raise ValueError('total size of new array must be unchanged')

  flat_indices = ncols * c.row + c.col
  new_row, new_col = divmod(flat_indices, shape[1])

  b = sps.coo_matrix((c.data, (new_row, new_col)), shape=shape)
  return b

def patchify(img, patch_shape):
  patch_side, y = patch_shape
  # img is (row, column, batch)
  img = img.transpose(2,0,1) 

  # Need batch to be first but want the patches to be ordered [[1, 2, 3]; [4, 5, 6]]
  img = np.ascontiguousarray(img) # won't make a copy if not needed
  # The right strides can be thought by:
  # 1) Thinking of `img` as a chunk of memory in C order
  # 2) Asking how many items through that chunk of memory are needed when indices
  #    i,j,k,l are incremented by one

  batch, image_side, image_side = img.shape
  image_dim = image_side**2
  shape = (batch, (image_side-patch_side+1), (image_side-y+1), patch_side, y)
  strides = img.itemsize*np.array([image_dim, image_side, 1, image_side, 1])

  return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
  # .reshape((image_side-patch_side+1)*(image_side-y+1), patch_side, y, batch)

def display_scales(t, G, Phi, patch_side=9, save=False, overcomplete=1, label=''):
  scales = len(Phi)

  print "Iteration " + str(t)

  for s in range(scales):
    (image_dim, total_neurons) = Phi[s].shape
    neurons = int(total_neurons/overcomplete)
    image_side = int(sqrt(image_dim))
  
    total_side = int(patch_side*sqrt(neurons)+sqrt(neurons)+1)

    image = -1*np.ones((total_side, overcomplete*total_side))
    for o in range(overcomplete):
      for i in range(int(sqrt(neurons))):
        for j in range(int(sqrt(neurons))):
          temp = reshape(Phi[s][:,int(o*neurons+i*sqrt(neurons)+j)],(image_side,image_side)).tolil()
          temp = temp[i:i+patch_side, j:j+patch_side].todense()
          temp = temp/np.max(np.abs(temp))
          start_row = int(i*patch_side+i+1)
          start_col =int(o*total_side+j*patch_side+j+1)
          image[start_row:start_row+patch_side, start_col:start_col+patch_side] = temp

      plt.imsave('./figures/level' + str(s) + '-' + str(label) + '-oc-' + str(overcomplete) + '-i' + str(image_side) + '-p' + str(patch_side) + 
        '-t' + str(t), image, cmap=cm.Greys_r)

  if save == True:
    f = file('./dictionaries/' + str(label) + '-oc-' + str(overcomplete) + '-i' + str(image_side) + '-p' + str(patch_side) +
      '-t' + str(t) + '-' + str(datetime.datetime.now()).split(' ', 1)[0], "wb")
    np.save(f, Phi)
    np.save(f, G)
    f.close()