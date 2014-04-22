import numpy as np
import array
import random
import os
import scipy.ndimage.filters
import sklearn.preprocessing

import laplacian_pyramid
import matplotlib.pyplot as plt

def extract_images(images='vanhateran', num_images=2000, image_dim=32*32, normalize=False, lowpass=True, bandpass=False, 
  whiten=False, patches_per_image=25, pad=False):
  """ Extract images from vanhateran dataset """

  image_side = int(np.sqrt(image_dim))
  border = 4

  IMAGES = np.zeros((num_images,image_dim))

  image_dir = '../../images/vanhateran/'
  filenames = filter( lambda f: not f.startswith('.'), os.listdir(image_dir))

  num_images = num_images/patches_per_image
  filenames = random.sample(filenames, num_images)

  for n,f in enumerate(filenames):
    img = preprocess('../../images/vanhateran/'+ f)
    (full_image_side, _) = np.shape(img)

    if bandpass == True:
      """Return the bottom layer of the Laplacian Pyramid to remove low range structure"""
      img = laplacian_pyramid.build(img, scales=2)[1]

    if lowpass == True:
      """Low pass filter and downsample highest octave to remove noise at highest frequencies"""
      img = scipy.ndimage.filters.gaussian_filter(img, sigma=1)
      img = img[::2, ::2]
      (full_image_side, _) = np.shape(img)

    for i in range(patches_per_image):
      r = int(border + np.ceil((full_image_side-image_side-2*border) * random.uniform(0,1)))
      c = int(border + np.ceil((full_image_side-image_side-2*border) * random.uniform(0,1)))
 
      IMAGES[n*patches_per_image+i] = img[r:r+image_side, c:c+image_side].reshape(image_dim) 

  if normalize == True:
    """Mean subtract and unit normlaize the image patches"""
    IMAGES = sklearn.preprocessing.scale(IMAGES, axis=1, with_std=False)
    IMAGES = sklearn.preprocessing.normalize(IMAGES, axis=1, norm='l2')

  if pad != False:
    IMAGES = IMAGES.reshape(num_images*patches_per_image, image_side, image_side)
    IMAGES = np.pad(IMAGES, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    IMAGES = IMAGES.reshape(num_images*patches_per_image, (image_side+2*pad)**2)


  if whiten != False:
    print "Whitening"
    IMAGES = IMAGES.dot(whiten)
      
  return IMAGES

def preprocess(filename):
  """ Function to read Van Hateran. Also crops to center 1024x1024 and takes log intensity"""
  R = 1024
  C = 1536
  extra = (C-R)/2
  with open(filename, 'rb') as handle:
    s = handle.read()

  arr = array.array('H', s)
  arr.byteswap()
  img = np.array(arr, dtype='uint16').reshape(R, C)
  img = img[:,extra-1:C-extra-1]

  return np.log(img+1) # log intensity

def binomial(mask_radius=1, dim=2):
  """ Returns a 1-D or 2-D Binomial Filter """
  length = 2*mask_radius+1
  K = scipy.misc.comb((length-1)*np.ones(length), np.arange(length))

  if dim == 1:
    return K/np.sum(np.abs(K))

  if dim == 2:
    K = np.outer(K,K)
    return K/np.sum(np.abs(K))

  return None
  

def whitening_matrix(X,fudge=10^50):
  # the matrix X should be observations-by-components
  Xcov = np.dot(X.T,X)
  # eigenvalue decomposition of the covariance matrix
  d,V = np.linalg.eigh(Xcov)

  # a fudge factor can be used so that eigenvectors associated with
  # small eigenvalues do not get overamplified.
  D = np.diag(1./np.sqrt(d+fudge))
  # whitening matrix
  W = np.dot(np.dot(V,D),V.T)
  # multiply by the whitening matrix
  X = np.dot(X,W)
 
  return d, X, W
  
