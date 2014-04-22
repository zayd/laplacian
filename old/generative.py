# Complete Laplacian transform of an image and uses generative model to reconstruct image. 
# Generative model: X = G_0 * a_0 + G_1 * a_i  + G_2 * a_2 

import random
import os
import array
import scipy.io
import numpy as np
from numpy import mean, std, ceil, mod, floor, dot, arange
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from math import sqrt
import datetime
import time

import preprocess

import skimage.transform as skt
from skimage.transform import pyramid_laplacian
from skimage.transform import pyramid_gaussian


#IMAGES = preprocess.extract_patches(images='vanhateran', num_images=100, image_dim=32*32)

def laplacianGenerative(base_image_dim, scales):
    """ 
    Gaussian generative matrices (as a tuple) of Laplacian pyramid in a csr format

    Assumes mode is 'zeros'
    Also zeros-out Gaussian kernels after x-sigma to save memory
    """

    sigma = (2*2)/6.0 # use sktransforms standard 2 * downscale / 6.0
    mask_radius = int(6*sigma) # (i.e. 6-sigma, values < 10e-7)

    def gaussian(image_side, sigma, mask_radius, center, normalize=False):
      """ 
      Generate a 2-d gaussian masked by mask in center. Returns a reshaped 1-d vector.
      Mask should be odd
      """
      x0, y0 = center
      
      #x = np.arange(-mask_radius, mask_radius+1, 1, float)
      #y = x[:,np.newaxis]
      #G_masked = np.exp(-(((x**2)+(y**2)) / (2*sigma**2)))

      G_masked = scipy.ndimage.filters.gaussian_filter(delta,sigma)

      if normalize == True:
        G_masked = G_masked/np.sum(np.abs(G_masked))

      G = sps.lil_matrix((image_side+2*mask_radius,image_side+2*mask_radius))
      # Add 2*mask_radius to all indices for padding

      G[x0:x0+2*mask_radius+1, y0:y0+2*mask_radius+1] = G_masked
      
      G = G[mask_radius:image_side+mask_radius,mask_radius:image_side+mask_radius]

      return G.reshape((1,image_side**2))

    G = range(scales)

    for s in range(scales):
      if s == 0:
        # l0 is Gaussians with sigma 1
        G[s] = sps.eye(base_image_dim,base_image_dim)

      else:
        base_image_side = int(sqrt(base_image_dim))
        image_dim = base_image_dim/(4**s)
        image_side = int(sqrt(image_dim))

        G[s] = sps.lil_matrix((image_dim, base_image_dim))

        for i in range(image_dim):
          G[s][i] = gaussian(base_image_side, (2**(s-1))*sigma, (2**(s-1))*mask_radius, 
            tuple(2**s * c for c in divmod(i,image_side)))  

        # csr format for fast computations
        G[s] = sps.csr_matrix(G[s])

    return G

def loopback(image, scales=3, A=None, pyramid=None):
  """
  Generate laplacian pyramid and return reconstructred matrix from generative matrices
  """
  base_image_side = image.shape[0]
  base_image_dim = base_image_side**2

  if pyramid == None:
    pyramid = tuple(pyramid_laplacian(image, downscale=2, max_layer=scales-1, mode='constant', order=5))
  
  if A == None:
    A = laplacianGenerative(base_image_dim, scales)

  recon = np.zeros((base_image_dim,1))

  for s in range(scales):
    image_dim = int(base_image_dim/4**s)
    recon += A[s].T.dot(pyramid[s].reshape((image_dim,1)))

  return recon
    