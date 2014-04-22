import numpy as np
from numpy import mean, std, ceil, mod, floor, dot, arange
import scipy.sparse as sps
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from math import sqrt


import skimage.transform as skt
from sklearn.preprocessing import normalize

import preprocess
import laplacian_pyramid

def learn(Phi=None, IMAGES=None, scales=3, patch_dim=9*9, 
    overcomplete=1, iterations=2000, batch=100, alpha=10, beta=0.998, lambdav=0.075, plot=False, save=False):
    # Alpha is powered based on scale. alpha_s = alpha**(scales-s-1)
    # 32x32 image size
    # 9x9 neurons. 1 pixel stride
    
    if IMAGES == None:
        num_images = 10000
        base_image_dim = 32*32
        # IMAGES = preprocess.extract_images(images='vanhateran', num_images=num_images, image_dim=image_dim)
    else:
        (base_image_dim, num_images) = IMAGES.shape

    base_image_side = int(sqrt(base_image_dim))
    patch_side = int(sqrt(patch_dim))
    pad = (patch_side-1)/2

    base_neurons = (base_image_side+(patch_side-1))**2

    if Phi == None:
        Phi = initialize(base_image_dim, patch_dim, scales)

    G = laplacian_pyramid.generative(base_image_dim, scales)

    for t in range(iterations+1):
      chosen_images = np.random.permutation(arange(0,num_images))[:batch]
      I = preprocess.extract_images(images='vanhateran', num_images=batch, image_dim=base_image_dim)

      A = sparsify(I.T, G, Phi, lambdav)
      R = reconstruct(G, Phi, A, base_image_dim)

      error = laplacian_pyramid.build(R - I, scales)
      dPhi = [sps.csc_matrix((base_image_dim/(4**(scales-s-1)), base_image_dim/(4**(scales-s-1)))) for s in range(scales)]

      for s in range(scales):
        image_dim = base_image_dim/(4**(scales-s-1))
        image_side = int(sqrt(image_dim))

        error[s] = error[s].reshape((image_side, image_side, batch))
        
        patches = patchify(np.pad(error[s], ((pad, pad), (pad, pad), (0, 0)), mode='constant'), (patch_side, patch_side), mode='2d')
        neurons = image_dim

        for n in range(neurons):
          print n
          update = np.dot(patches[n], A[s][n].T)
          print localize(update, image_side, patch_side, divmod(n, image_side)) 
          dPhi[s][:, n] = localize(update, image_side, patch_side, divmod(n, image_side)) 

      for s in range(s):
        Phi[s] = Phi[s] + alpha * dPhi[s]
        normalize(Phi[s], norm='l1', axis=0, copy=False)

      if mod(t,10):
        print t
        display()


def reconstruct(G, Phi, A, base_image_dim, plot=False):
    scales = len(Phi)
    (_, patch_dim) = Phi[0].shape
    (_, batch) = A[0].shape

    R = np.zeros((base_image_dim, batch))

    for s in range(scales):
        image_dim = int(base_image_dim/4**(scales-1-s))
        R += G[s].dot(Phi[s].dot(A[s]))

    if plot == True:
        for s in range(scales):
            plt.imsave('test' + str(s), R)

    # Return C style array
    return R.T

def sparsify(I, G, Phi, lambdav, iterations=150, eta=0.1):
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

  scales = len(G)
  (base_image_dim, batch) = I.shape

  # Gamma = <(G_0*Phi_0 + G_1*Phi_1 + G_2*Phi_2), (G_0*Phi_0 + G_1*Phi_1 + G_2*Phi_2)>
  # b = <(G_0*Phi_0 + G_1*Phi_1 + G_2*Phi_2), I>

  M = sps.hstack([G[s]*Phi[s] for s in range(scales)]).tocsr()
  total_neurons = M.shape[1]
  Gamma = M.T * M - sps.eye(total_neurons, total_neurons)
  Gamma = Gamma.tocsr()

  b = M.T.dot(I)

  u = np.zeros((M.shape[1],batch))

  l = 0.5 * np.max(np.abs(b), axis=0)
  a = g(u,l, 'soft')
  olda = a

  t = 0
  while (t < iterations+1) or (np.sqrt(np.sum((olda-a)**2)) > 10e5):
    olda = a
    u = eta * (b-Gamma.dot(a)) + (1-eta) * u
    a = g(u,l, 'soft')
    l = 0.95 * l
    l[l < lambdav] = lambdav

    # print np.sum((a-olda)**2)
    t += 1

  print np.sum((a-olda)**2)

  A = range(scales)
  for s in range(scales):
    neurons = Phi[s].shape[1]
    A[s] = a[:neurons,:]
    a = a[neurons:,:]

  return A

def initialize(base_image_dim, patch_dim, scales):
  """ Initialize sparse Phi matrix with Gaussian random noise """
  Phi = range(scales)
  base_image_side = int(sqrt(base_image_dim))
  patch_side = int(sqrt(patch_dim))
  pad = patch_side-1
  mask_radius = pad/2

  for s in range(scales):
    image_side = base_image_side/(2**(scales-(s+1)))
    pad_image_side = image_side + pad
    pad_image_dim = pad_image_side**2

    neurons = image_side**2

    Phi[s] = sps.lil_matrix((pad_image_dim, neurons))

    for n in range(neurons):
      Phi[s][:,n] = localize(np.random.randn(patch_side, patch_side), pad_image_side, 
        patch_side, (s + mask_radius for s in divmod(n, image_side)))

    Phi[s] = sps.csc_matrix(Phi[s])
    normalize(Phi[s], norm='l1', axis=0, copy=False)

  return Phi

def localize(kernel, image_side, patch_side, center):
    """ Localize smaller kernel in larger sparse matrix """
    x0, y0 = center

    mask_radius = (patch_side-1)/2
    K = sps.lil_matrix((image_side+2*mask_radius, image_side+2*mask_radius))
    K[x0:x0+2*mask_radius+1, y0:y0+2*mask_radius+1] = kernel

    K = K[mask_radius:image_side+mask_radius, mask_radius:image_side+mask_radius]

    return K.reshape((image_side**2, 1))

def patchify(img, patch_shape, mode='1d'):
    patch_side, y = patch_shape
    img = np.ascontiguousarray(img) # won't make a copy if not needed

    if mode == '1d': 
      # Images in (image_dim, batch) format 
      image_dim, batch = img.shape
      image_side = int(sqrt(image_dim))
      shape = (patch_side, y, (image_side-patch_side+1), (image_side-y+1), batch) # number of patches, patch_shape

      # The right strides can be thought by:
      # 1) Thinking of `img` as a chunk of memory in C order
      # 2) Asking how many items through that chunk of memory are needed when indices
      #    i,j,k,l are incremented by one
      strides = img.itemsize*np.array([image_side, 1, image_side, 1, image_dim])
      return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides).reshape(
        patch_side*y, (image_side-patch_side+1)*(image_side-y+1), batch)

    elif mode == '2d':
      image_side, image_side, batch = img.shape
      image_dim = image_side**2
      shape = ((image_side-patch_side+1), (image_side-y+1), patch_side, y,  batch)
      strides = img.itemsize*np.array([1, image_side, image_side, 1,  image_dim])

      return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides).reshape((image_side-patch_side+1)*(image_side-y+1), patch_side, y, batch)

def display(t, Phi, base_image_side, save=False, overcomplete=1):
  scales = len(Phi)
  
  patch_side = sqrt(patch_dim)
  print "Iteration " + str(t)

  for s in range(scales):
    (patch_dim, neurons) = Phi[s].shape
    patch_side = int(sqrt(patch_dim))
    image = -1*np.ones((patch_side*np.sqrt(neurons)+sqrt(neurons)+1,patch_side*sqrt(neurons)+sqrt(neurons)+1))
    for i in range(int(sqrt(neurons))):
      for j in range(int(sqrt(neurons))):
        temp = np.reshape(Phi[s][:,i*sqrt(neurons)+j],(patch_side,patch_side))-np.min(Phi[s][:,i*int(sqrt(neurons))+j])
        temp = temp/np.max(np.abs(Phi[s][i*sqrt(neurons)+j,:]))
        temp = 2*(temp-0.5)
        image[i*patch_side+i+1:i*patch_side+patch_side+i+1,j*patch_side+j+1:j*patch_side+patch_side+j+1] = temp

    plt.imsave('./figures/level' + str(s) + '-t' + str(t), image, cmap=cm.Greys_r)

    if save == True:
      np.save('./dictionaries/oc' + str(overcomplete) + '-s' + str(scales) +
        '-t' + str(t) + '-' + str(datetime.datetime.now()), Phi)

"""
def get_indices(image_side, padding, patch_side, scales):
  np.arange((image_side+2*padding)**2)

  indices = range(scales)
  neurons = image_side**2

  for s in range(scales):
    indices[s] = [[], np.arange(neurons)]
    indices[s][0] = patchify([:, None], (patch_side, patch_side))[:,:,0]

  return indices
"""



