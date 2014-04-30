import scipy.signal
import scipy.sparse as sps
import scipy.ndimage as spn
import numpy as np
from numpy import mean, std, ceil, mod, floor, dot, arange
import math
from math import sqrt

def build(I, scales, filter_radius=2):
  """ Build a laplacian pyramid 
  I: images (batch x image_side x image_side)
  scales: levels in pyramid
  filter_radius: size of filter to use (std. for gaussian - not the same as radius)

  Returns list of numpy arrays where each element in list is scale
  (scale) x (batch x image_side x image_side)
  """ 
  if scales == 1:
    pyramid = list()
    pyramid.append(I)
    return pyramid
  else:
    G1 = shrink(I)
    L0 = I-expand(G1)

    pyramid = build(G1, scales-1)
    pyramid.append(L0)

    return pyramid

def reconstruct(pyramid):
  if len(pyramid) == 1:
    return pyramid[0]
  else:
    pyramid[1] = pyramid[1] + expand(pyramid[0])
    return reconstruct(pyramid[1:])

def shrink(image, downscale=2, filter_radius=1):
  image = blur(image, mask_radius=filter_radius)

  return image[:, ::downscale,::downscale]

def expand(image, upscale=2, filter_radius=1):
  (batch, image_x ,image_y) = image.shape
  image_expanded = np.zeros((batch, upscale*image_x, upscale*image_y))
  image_expanded[:, ::upscale, ::upscale] = image

  return blur(image_expanded, mask_radius=filter_radius)

def blur(image, kernel_type='gaussian', mask_radius=1):
  if kernel_type == 'gaussian':
    # Do it as a seperable 1d convolution (but not along batch axis)
    image = spn.gaussian_filter1d(image, sigma=mask_radius, axis=1, mode='constant')
    image = spn.gaussian_filter1d(image, sigma=mask_radius, axis=2, mode='constant')
    return image

def generative(base_image_side, patch_side, scales, kernel_type='binomial', base_mask_radius=2):
  """ Laplacian generative matrices of Laplacian pyramid in a csr format """

  def localize(kernel, image_side, center, mask_radius):
    x0, y0 = center

    K = sps.lil_matrix((image_side+2*mask_radius, image_side+2*mask_radius))
    K[x0:x0+2*mask_radius+1, y0:y0+2*mask_radius+1] = kernel

    K = K[mask_radius:image_side+mask_radius, mask_radius:image_side+mask_radius]
    return K.reshape(((image_side)**2, 1))

  G = range(scales)
  base_image_dim = base_image_side**2
  pad = (patch_side-1)/2

  if kernel_type == 'binomial':
    kernel = binomial(base_mask_radius)

  for s in range(scales):
      if s == 0:
        # l0 is Gaussians with sigma 1
        G[s] = sps.eye((base_image_side+2*pad)**2, (base_image_side+2*pad)**2)
        # Set the edges to be zero
        #zero_indices = _padding_indices(base_image_side, pad)
      else:
        image_dim = base_image_dim/(4**s)
        image_side = int(sqrt(image_dim))
        zero_indices = _padding_indices(image_side, pad)
        mask_radius = (kernel.shape[0]-1)/2

        G[s] = sps.lil_matrix(((base_image_side+2*pad)**2, (image_side+2*pad)**2))

        j = 0
        for i in range((image_side+2*pad)**2):
          if i == zero_indices[0]:
            #print "caught i: " + str(i)
            zero_indices = zero_indices[1:]

          else:
            # scale kernel by 4 to not lose any power in 2D interpolation
            G[s][:,i] = localize(4*kernel, base_image_side+2*pad, 
             tuple(2**s * c + pad for c in divmod(j, image_side)), mask_radius=mask_radius)
            j += 1

        # convolve kernel for next scale
        expanded_kernel = np.zeros((2*kernel.shape[0], 2*kernel.shape[1]))
        expanded_kernel[::2, ::2] = kernel
        expanded_kernel = scipy.signal.convolve2d(expanded_kernel, binomial(mask_radius), mode='full')
        kernel = expanded_kernel[:-1, :-1]

        # csr format for fast computations
        G[s] = sps.csr_matrix(G[s])

  return G[::-1] # Reverse to match laplacian pyramid data structure

def _padding_indices(image_side, pad):
  I = np.ones((image_side, image_side))
  I = np.pad(I, ((pad, pad), (pad, pad)), mode='constant').reshape((image_side+2*pad)**2)
  return np.where(I == 0)[0]

def _loopback(image, patch_side, scales=3, G=None, pyramid=None):
  """
  Build laplacian pyramid using build and return reconstructred image from generative matrices
  """
  base_image_side = image.shape[0]
  base_image_dim = base_image_side**2

  if pyramid == None:
    pyramid = build(image, scales=scales)
  
  if G == None:
    G = generative(base_image_side, patch_side, scales)

  R = np.zeros((base_image_dim,1))

  for s in range(scales):
    image_dim = int(base_image_dim/4**(scales-1-s))
    R += G[s].dot(pyramid[s].reshape((image_dim,1)))

  return R
