# Implements sparse coding using tiling on a single layer
import numpy as np
from numpy import mean, std, ceil, mod, floor, dot, arange
import scipy.sparse as sps
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from math import sqrt
import time

import skimage.transform as skt
import sklearn.preprocessing as skp

import preprocess
import fista

def learn(Phi=None, image_dim=32*32, patch_dim=9*9, normalize=True, bandpass=False,
    overcomplete=1, iterations=4000, batch=100, alpha=0.8, beta=0.9995, gamma=0.95, lambdav=0.05, plot_every=50, label=None):
    # Main learning loop

    patch_side = int(sqrt(patch_dim))
    image_side = int(sqrt(image_dim))
    pad = (patch_side-1)/2
    indices = all_indices(image_side, patch_side, overcomplete)

    if Phi == None:
        Phi = initialize(image_side, patch_side, overcomplete)

    neurons = overcomplete*image_dim
    old_dPhi = np.zeros((neurons, patch_dim))

    for t in range(iterations+1):
      I = preprocess.extract_images(images='vanhateran', num_images=batch, image_dim=image_dim, normalize=normalize, bandpass=bandpass)
      I = I.T
      I = np.pad(I.reshape(image_side, image_side, batch), ((pad, pad), (pad, pad), (0, 0)), mode='constant')
      I = I.reshape((image_side+2*pad)*(image_side+2*pad), batch)
      
      # A = sparsify(I, Phi, lambdav)
      A = fista.fista(I, Phi, lambdav, max_iterations=10*overcomplete, display=True)
      R = reconstruct(Phi, A)

      error = I - R
      error = error.reshape(image_side+2*pad, image_side+2*pad, batch)
      # TO DO: set error on paddings to 0
      error = patchify(error, (patch_side, patch_side))
      error = error.reshape(batch,neurons/overcomplete,patch_dim)
      error = np.tile(error, (1, overcomplete, 1)) # Repeat for OC

      dPhi = error.transpose(1,2,0) * A[:, None, :]
      dPhi = dPhi.sum(axis=2)
      dPhi = (1-gamma)*dPhi + gamma*old_dPhi
      old_dPhi = dPhi 

      #print "Old Objective: " + str(np.sum((I-R)**2) + lambdav*np.sum(np.abs(A)))
      Phi = Phi.tolil()
      Phi[indices[0], indices[1]] = Phi[indices[0], indices[1]] + (alpha/float(batch)) * dPhi.T
      Phi = Phi.tocsc()
      skp.normalize(Phi, norm='l2', axis=0, copy=False)

      A = fista.fista(I, Phi, lambdav, max_iterations=10*overcomplete, display=False)
      R = reconstruct(Phi, A)
      print "New Objective: " + str(np.sum((I-R)**2) + lambdav*np.sum(np.abs(A)))

      # Armajillo's Rule
      alpha = alpha * beta

      if t % plot_every == 0:
        display(t, Phi, save=True, patch_dim=patch_dim, overcomplete=overcomplete, label=label)

def learn_conv(Phi=None, scales=3, image_dim=32*32, patch_dim=9*9, whiten=True,
    overcomplete=1, iterations=2000, batch=100, alpha=400, beta=0.9995, gamma=0.95, lambdav=0.05, plot=False, save=False):
    # Main learning loop

    label = 'conv'

    patch_side = int(sqrt(patch_dim))
    image_side = int(sqrt(image_dim))
    pad = (patch_side-1)/2
    indices = all_indices(image_side, patch_side, overcomplete)

    if Phi == None:
        Phi = initialize(image_side, patch_side, overcomplete, convolutional=True)

    neurons = overcomplete*image_dim
    #old_dPhi = np.zeros((neurons, patch_dim))

    if whiten == True:
      label = label + '-whitened'
      print "Whitening"
      I = preprocess.extract_images(images='vanhateran', num_images=50000, image_dim=image_dim)
      (_, W) = preprocess.whitening_matrix(I)

    for t in range(iterations+1):
      I = preprocess.extract_images(images='vanhateran', num_images=batch, image_dim=image_dim, whiten=W)
      I = I.T
      I = np.pad(I.reshape(image_side, image_side, batch), ((pad, pad), (pad, pad), (0, 0)), mode='constant')
      I = I.reshape((image_side+2*pad)*(image_side+2*pad), batch)

      # A = sparsify(I, Phi, lambdav)
      A = fista.fista(I, Phi[0::], lambdav, max_iterations=50)
      R = reconstruct(Phi, A)

      error = I - R
      error = error.reshape(image_side+2*pad, image_side+2*pad, batch)
      e = error
      # TO DO: set error on paddings to 0
      error = patchify(error, (patch_side, patch_side))
      error = error.reshape(batch,neurons/overcomplete,patch_dim)
      error = np.tile(error, (1, overcomplete, 1)) # Repeat for OC

      dPhi = error.transpose(1,2,0) * A[:, None, :]
      dPhi = dPhi.sum(axis=2) # Sum over batch
      dPhi = sum_chunk(dPhi, neurons/overcomplete, axis=0)
      dPhi = dPhi/float(neurons/overcomplete) #normalize
      dPhi = dPhi.repeat(neurons/overcomplete, axis=0)

      #dPhi = (1-gamma)*dPhi + gamma*old_dPhi
      #old_dPhi = dPhi 

      # print "Old Objective: " + str(np.sum((I-R)**2) + lambdav*np.sum(np.abs(A)))
      Phi = Phi.tolil()
      Phi[indices[0], indices[1]] = Phi[indices[0], indices[1]] + (alpha/float(batch)) * dPhi.T
      Phi = Phi.tocsc()
      skp.normalize(Phi, norm='l2', axis=0, copy=False)

      #A = sparsify(I, Phi, lambdav)
      # A = fista.fista(I, Phi, lambdav, max_iterations=50)
      # R = reconstruct(Phi, A)
      # print "New Objective: " + str(np.sum((I-R)**2) + lambdav*np.sum(np.abs(A)))

      # Armajillo's Rule
      alpha = alpha * beta

      if t % 50 == 0:
        display(t, Phi, save=True, patch_dim=patch_dim, overcomplete=overcomplete, label=label)

def learn_scales(Phi=None, scales=2, image_dim=32*32, patch_dim=9*9, normalize=True, bandpass=False,
    overcomplete=1, iterations=4000, batch=100, alpha=400, beta=0.9995, gamma=0.95, lambdav=0.05, plot_every=50, label=None):
    # Main learning loop

    patch_side = int(sqrt(patch_dim))
    image_side = int(sqrt(image_dim))
    pad = (patch_side-1)/2
    indices = all_indices(image_side, patch_side, overcomplete)

    if Phi == None:
        Phi = initialize(image_side, patch_side, overcomplete)

    neurons = overcomplete*image_dim
    old_dPhi = np.zeros((neurons, patch_dim))

    for t in range(iterations+1):
      I = preprocess.extract_images(images='vanhateran', num_images=batch, image_dim=image_dim, normalize=normalize, bandpass=bandpass)
      I = I.T
      I = np.pad(I.reshape(image_side, image_side, batch), ((pad, pad), (pad, pad), (0, 0)), mode='constant')
      I = I.reshape((image_side+2*pad)*(image_side+2*pad), batch)
      
      # A = sparsify(I, Phi, lambdav)
      A = fista.fista(I, Phi, lambdav, max_iterations=10*overcomplete, display=True)
      R = reconstruct(Phi, A)

      error = I - R
      error = error.reshape(image_side+2*pad, image_side+2*pad, batch)
      # TO DO: set error on paddings to 0
      error = patchify(error, (patch_side, patch_side))
      error = error.reshape(batch,neurons/overcomplete,patch_dim)
      error = np.tile(error, (1, overcomplete, 1)) # Repeat for OC

      dPhi = error.transpose(1,2,0) * A[:, None, :]
      dPhi = dPhi.sum(axis=2)
      dPhi = (1-gamma)*dPhi + gamma*old_dPhi
      old_dPhi = dPhi 

      print "Old Objective: " + str(np.sum((I-R)**2) + lambdav*np.sum(np.abs(A)))
      Phi = Phi.tolil()
      Phi[indices[0], indices[1]] = Phi[indices[0], indices[1]] + (alpha/float(batch)) * dPhi.T
      Phi = Phi.tocsc()
      skp.normalize(Phi, norm='l2', axis=0, copy=False)

      A = fista.fista(I, Phi, lambdav, max_iterations=10*overcomplete, display=False)
      R = reconstruct(Phi, A)
      print "New Objective: " + str(np.sum((I-R)**2) + lambdav*np.sum(np.abs(A)))

      # Armajillo's Rule
      alpha = alpha * beta

      if t % plot_every == 0:
        display(t, Phi, save=True, patch_dim=patch_dim, overcomplete=overcomplete, label=label)

def reconstruct(Phi, A):
    return Phi.dot(A)

def sparsify(I, Phi, lambdav, iterations=75, eta=0.1):
  def g(u, theta, thresh_type='soft'):
    """
    LCA threshold function
    u: coefficients
    theta: threshold value
    """
    if thresh_type == 'hard':
        a = u;
        a[np.abs(a) < theta] = 0
        return a
    elif thresh_type == 'soft':
        a = np.abs(u)-theta
        a[a<0] = 0
        a = np.sign(u)*a
        return a

  (image_dim, batch) = I.shape
  # Gamma = <(G_0*Phi_0 + G_1*Phi_1 + G_2*Phi_2), (G_0*Phi_0 + G_1*Phi_1 + G_2*Phi_2)>
  # b = <(G_0*Phi_0 + G_1*Phi_1 + G_2*Phi_2), I>
  neurons = Phi.shape[1]
  Gamma = Phi.T * Phi - sps.eye(neurons, neurons)
  # TO DO: remove dot product between basis functions in corner by setting Phi to 0 in corners for All basis functions
  Gamma = Gamma.tocsr()

  b = Phi.T.dot(I)
  u = np.zeros((neurons,batch))

  l = 0.5 * np.max(np.abs(b), axis=0)
  a = g(u,l, 'soft')
  olda = a

  t = 0

  t1 = time.time()
  while (t < iterations+1): # or (np.sqrt(np.sum((olda-a)**2)) > 10e5):
    olda = a
    u = eta * (b-Gamma.dot(a)) + (1-eta) * u

    a = g(u,l, 'soft')
    l = 0.95 * l
    l[l < lambdav] = lambdav

    # print np.sum((a-olda)**2)
    t += 1
  print time.time() - t1
  #print np.sum((a-olda)**2)
  print "Avg. L1 Norm: " + str(np.sum(np.abs(a))/float(batch))

  return a

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
  skp.normalize(Phi, norm='l2', axis=0, copy=False)

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

# Helper functions
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
  # Need batch to be first but want the patches to be ordered [[1 2 3][4 5 6]
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

def sum_chunk(x, chunk_size, axis=-1):
    """Sum chunks of matrix"""
    shape = x.shape
    if axis < 0:
        axis += x.ndim
    shape = shape[:axis] + (-1, chunk_size) + shape[axis+1:]
    x = x.reshape(shape)
    return x.sum(axis=axis+1)

def display(t, Phi, patch_dim=9*9, save=False, overcomplete=1, label=''):
  (image_dim, total_neurons) = Phi.shape
  neurons = total_neurons/overcomplete
  image_side = int(sqrt(image_dim))
  patch_side = int(sqrt(patch_dim))
  print "Iteration " + str(t)

  image = -1*np.ones((overcomplete*(patch_side*np.sqrt(neurons)+sqrt(neurons)+1),patch_side*sqrt(neurons)+sqrt(neurons)+1))
  for o in range(overcomplete):
    for i in range(int(sqrt(neurons))):
      for j in range(int(sqrt(neurons))):
        temp = reshape(Phi[:,o*neurons+i*sqrt(neurons)+j],(image_side,image_side))
        temp = temp.tolil()
        temp = temp[i:i+patch_side, j:j+patch_side].todense()
        temp = temp/np.max(np.abs(temp))
        temp = 2*(temp-0.5)
        image[int(o*(patch_side*np.sqrt(neurons)+sqrt(neurons)+1)+i*patch_side+i+1):int(o*(patch_side*np.sqrt(neurons)+sqrt(neurons)+1)+i*patch_side+patch_side+i+1),j*patch_side+j+1:j*patch_side+patch_side+j+1] = temp

    plt.imsave('./figures/' + str(label) + '-oc-' + str(overcomplete) + '-i' + str(image_side) + '-p' + str(patch_side) + '-t' + str(t), image, cmap=cm.Greys_r)

  if save == True:
    np.save('./dictionaries/' + str(label) + '-oc-' + str(overcomplete) + '-i' + str(image_side) + '-p' + str(patch_side) +
      '-t' + str(t) + '-' + str(datetime.datetime.now()), Phi)



