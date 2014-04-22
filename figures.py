#import laplacian
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import fista
import preprocess
import single

import scales

def figures(Phi=None):

  IMAGES = laplacian.laplacian(image_dim=32*32, images='vanhateran', scales=3, normalize=True, num_patches=10000)
  (base_image_dim, num_images) = IMAGES[0].shape

  if Phi == None:
    Phi = laplacian.learn(IMAGES=IMAGES, iterations=1000, patch_dim=8*8, alpha=100, save=True, batch=100)

  # Example reconstruction
  fig1, ax = plt.subplots(2)
  reconstruction = laplacian.reconstruct(Phi,IMAGES[0],base_image_dim)
  ax[0].imshow(laplacian.laplacian_reconstruct(IMAGES[0]))
  ax[1].imshow(reconstruction)

  fig2, ax = plt.subplots(1)


def snr():
  # Function that plots reconstruction error by sparsity
  sparsity = [0.001, 0.01, 0.1, 1]
  for s in sparsity:
    print s

def mse_vs_sparsity(batch=100, image_dim=32*32, patch_dim=9*9):
  patch_side = int(np.sqrt(patch_dim))
  image_side = int(np.sqrt(image_dim))
  pad = (patch_side-1)/2

  I = preprocess.extract_images(images='vanhateran', num_images=batch, image_dim=image_dim)
  I = I.T
  I = np.pad(I.reshape(image_side, image_side, batch), ((pad, pad), (pad, pad), (0, 0)), mode='constant')
  I = I.reshape((image_side+2*pad)*(image_side+2*pad), batch)

  """
  Phi_conv_oc2_l005 = np.load('/Users/zayd/Dropbox/Code/multiscale/laplacian/dictionaries/conv-oc-2-i40-p9-t1900-2014-02-04 09:39:29.570434.npy').item()
  Phi_conv_oc4_l005 = np.load('/Users/zayd/Dropbox/Code/multiscale/laplacian/dictionaries/conv-oc-4-i40-p9-t400-2014-02-04 15:47:39.352604.npy').item()

  Phi_oc_l005 = np.load('/Users/zayd/Dropbox/Code/multiscale/laplacian/dictionaries/oc1-i40-p9-t1000-2014-02-03 22:54:32.030729.npy').item()
  Phi_oc2_l005 = np.load('/Users/zayd/Dropbox/Code/multiscale/laplacian/dictionaries/oc2-i40-p9-t1000-2014-02-04 00:51:06.116929.npy').item()
  Phi_oc4_l005 = np.load('/Users/zayd/Dropbox/Code/multiscale/laplacian/dictionaries/oc4-i40-p9-t1800-2014-02-04 16:00:06.398794.npy').item()
=======
  Phi_conv_oc2_l005 = np.load('/Users/zayd/Dropbox/Code/multiscale/laplacian2/dictionaries/conv-oc-2-i40-p9-t1900-2014-02-04 09:39:29.570434.npy').item()
  Phi_conv_oc4_l005 = np.load('/Users/zayd/Dropbox/Code/multiscale/laplacian2/dictionaries/conv-oc-4-i40-p9-t400-2014-02-04 15:47:39.352604.npy').item()

  Phi_oc_l005 = np.load('/Users/zayd/Dropbox/Code/multiscale/laplacian2/dictionaries/oc1-i40-p9-t1000-2014-02-03 22:54:32.030729.npy').item()
  Phi_oc2_l005 = np.load('/Users/zayd/Dropbox/Code/multiscale/laplacian2/dictionaries/oc2-i40-p9-t1000-2014-02-04 00:51:06.116929.npy').item()
  Phi_oc4_l005 = np.load('/Users/zayd/Dropbox/Code/multiscale/laplacian2/dictionaries/oc4-i40-p9-t1800-2014-02-04 16:00:06.398794.npy').item()
>>>>>>> 6764613a3f3ef85b2fb61beb9079b57bba057ece

  A_conv_oc2_l005 = fista.fista(I, Phi_conv_oc2_l005, lambdav=0.05, max_iterations=100)
  A_conv_oc4_l005 = fista.fista(I, Phi_conv_oc4_l005, lambdav=0.05, max_iterations=100)

  A_oc2_l005 = fista.fista(I, Phi_oc2_l005, lambdav=0.05, max_iterations=100)
  A_oc4_l005 = fista.fista(I, Phi_oc4_l005, lambdav=0.05, max_iterations=100)
 
  R_conv_oc2_l005 = single.reconstruct(Phi_conv_oc2_l005, A_conv_oc2_l005)
  R_conv_oc4_l005 = single.reconstruct(Phi_conv_oc4_l005, A_conv_oc4_l005)

  R_oc2_l005 = single.reconstruct(Phi_oc2_l005, A_oc2_l005)
  R_oc4_l005 = single.reconstruct(Phi_oc4_l005, A_oc4_l005)

  mse_conv_oc2_l005 = np.sum((I-R_conv_oc2_l005)**2)/float(batch)
  mse_conv_oc4_l005 = np.sum((I-R_conv_oc4_l005)**2)/float(batch)
  mse_oc2_l005 = np.sum((I-R_oc2_l005)**2)/float(batch)
  mse_oc4_l005 = np.sum((I-R_oc4_l005)**2)/float(batch)

  l1_conv_oc2_l005 = np.sum(np.abs(A_conv_oc2_l005))/float(batch)
  l1_conv_oc4_l005 = np.sum(np.abs(A_conv_oc4_l005))/float(batch)

  l1_oc2_l005 = np.sum(np.abs(A_oc2_l005))/float(batch)
  l1_oc4_l005 = np.sum(np.abs(A_oc4_l005))/float(batch)

  x = [l1_conv_oc2_l005, l1_conv_oc4_l005, l1_oc2_l005, l1_oc4_l005]
  y = [mse_conv_oc2_l005, mse_conv_oc4_l005, mse_oc2_l005, mse_oc4_l005]

  y = [s/float(image_dim) for s in y]
  """

  plt.scatter(x,y)
  plt.xlabel('Sparsity (l1 Norm)')
  plt.ylabel('MSE (per pixel)')
  plt.show()

  return (x,y)

def mse_vs_sparsity(file_names, batch=100, image_dim=32*32):
  
  dictionaries = []

  for f in file_names:
    dictionaries.append(np.load(f).item())


  for d in dictionaries:
    a.append(scales.inference())

  plt.scatter(x,y)
  plt.show()


  
