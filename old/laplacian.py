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

import preprocess

import skimage.transform as skt
from skimage.transform import pyramid_laplacian
from skimage.transform import pyramid_gaussian

def laplacian(image_dim=32*32, scales=3, normalize=False, plot=False, images='vanhateran', num_patches=4000):
	# Generate Laplacian pyramids from data set
	image_side=math.sqrt(image_dim)

	if images == 'vanhateran':
		IMAGES = preprocess.extract_patches(images='vanhateran', num_images=num_patches, image_dim=image_dim)

	elif images == 'berkeley':
		IMAGES = np.zeros((image_side,image_side,num_patches))
		FIMAGES = scipy.io.loadmat('../../images/IMAGES.mat')
		FIMAGES = FIMAGES['IMAGES']
		(full_image_size, full_image_size, full_num_patches) = FIMAGES.shape
		BUFF = 4

		for i in range(num_patches):
			imi = np.floor(full_num_patches * random.uniform(0,1))
			r = BUFF + np.ceil((full_image_size-image_side-2*BUFF) * random.uniform(0,1))
			c = BUFF + np.ceil((full_image_size-image_side-2*BUFF) * random.uniform(0,1))
			IMAGES[i,:,:] = FIMAGES[r:r+image_side, c:c+image_side, imi]

	elif images == 'cifar':
		def unpickle(file):
		 # Open CIFAR files
			import cPickle
			fo = open(file, 'rb')
			dict = cPickle.load(fo)
			fo.close()
			return dict
		data = 'data_batch_1'
		cifar = unpickle('./data/cifar-10/' + data)
		IMAGES = cifar['data'].T
		IMAGES = IMAGES - np.mean(IMAGES, axis=0) # Mean subtract

	pyramids = []
	I = tuple([np.zeros((num_patches, image_dim/(2*2)**s)) for s in range(scales)])
	#I = np.empty((image_dim, scales, num_patches))

	for i in range(num_patches):
		pyramids.append(tuple(pyramid_laplacian(IMAGES[i,:,:], downscale=2, max_layer=scales-1)))

		if normalize == True:
			for s in range(scales):
				temp = pyramids[i][s].reshape((image_dim/(2*2)**s))
				norm = float(sqrt(np.sum(temp**2)))
				if norm != 0:
					I[s][i,:image_dim/(2*2)**s] = temp/norm
				else:
					I[s][i,:image_dim/(2*2)**s] = temp

		else:
			for s in range(scales):
				I[s][i,:image_dim/(2*2)**s] = pyramids[i][s].reshape((image_dim/(2*2)**s))

	if plot:
		fig, ax = plt.subplots(scales+1)
		ax[0].imshow(IMAGES[1,:,:], cmap='Greys', vmin=np.min(IMAGES[1,:,:]), vmax=np.max(IMAGES[1,:,:]), interpolation='nearest')
		for i in range(1,4):
			ax[i].imshow(I[1,:,i-1].reshape(image_side,image_side), vmin=np.min(I[1,:,i-1]), vmax=np.max(I[1,:,i-1]),
				cmap='Greys',  interpolation='nearest')

	return I

def laplacian_reconstruct(pyr):
	# Takes tuple of Laplacian pyramid levels and reconstructs original images
	if len(pyr) == 1:
		return pyr[0]
	else:
		return pyr[0] + skt.pyramid_expand(laplacian_reconstruct(pyr[1:]), upscale=2, sigma=10, mode='constant', cval=0.0)

def neuron_indices(neurons, patch_dim):
	for i in range(neurons):
		loc = i % num_neurons_tile
		j = floor(loc/(image_side-(patch_side-1))) % (image_side-(patch_side-1)) 	# Current row
		k = loc % (image_side-(patch_side-1))		# Current col

def neuron_mappings(Phi, overcomplete=1):
	# Takes dictionary and returns mapping between total neurons and the indices at each level.
	# Returns tuple of lists where each element points to indice of each basis function
	scales = len(Phi)
	(base_neurons, patch_dim) = Phi[0].shape
	base_neurons = base_neurons/overcomplete # Number of neurons at base. Overcompleteness separate

	mappings = [[] for s in range(scales)]
	for s in range(scales):
		for o in range(overcomplete):
			neurons = Phi[s].shape[0]/overcomplete
			ratio = ceil(sqrt(base_neurons)/float(sqrt(neurons)))
			for i in range(int(sqrt(neurons))):
				for j in range(int(sqrt(neurons))):
					mappings[s].append(o*base_neurons+ratio*(i*sqrt(base_neurons)+j))
	return mappings


def learn(Phi=None, IMAGES=None, patch_dim=8*8,
	overcomplete=1, iterations=2000, batch=100, alpha=10, beta=0.998, lambdav=0.075, plot=False, save=False):
	# Alpha is powered based on scale. alpha_s = alpha**(scales-s-1)
	# 32x32 image size
	# 8x8 neurons. 1 pixel offset

	if IMAGES == None:
		IMAGES = laplacian(image_dim=32*32, images='vanhateran', scales=3, num_patches=1000)
	# IMAGES = np.load(images_file)
	scales = len(IMAGES)
	(num_patches, base_image_dim) = IMAGES[0].shape
	base_image_side = math.sqrt(base_image_dim)
	patch_side = math.sqrt(patch_dim)

	base_neurons = (base_image_side-(patch_side-1))**2
	#neurons = int(overcomplete*)

	# from patch_dim,neurons,scales -> scales,neurons,patch_dim
	if Phi == None:
		Phi = [np.random.randn(overcomplete*((base_image_side/2**s)-patch_side+1)**2, patch_dim) for s in range(scales)]

	mappings = neuron_mappings(Phi, overcomplete=overcomplete)

	# Normalize bases
	for s in range(scales):
		Phi[s] = Phi[s]/np.sqrt(np.sum(Phi[s]**2, axis=1))[:,None]

	for t in range(iterations+1):
		chosen_images = np.random.permutation(arange(0,num_patches))[:batch]
		I = tuple([IMAGES[s][chosen_images,:] for s in range(scales)])

		# Sparsify. Compute sparse coefficients
		A = sparsify(I, Phi, mappings, lambdav)
		# Reconstruct each image in batch
		R = reconstruct(Phi, A, mappings, base_image_dim)

		if plot:
			fig, ax = plt.subplots(scales)
			fig2, ax2 = plt.subplots(scales)
			for s in range(scales):
				ax[s].imshow(R[s][0,:,:], interpolation='nearest')
				ax2[s].imshow(IMAGES[s][0,:].reshape(base_image_side/2**s,base_image_side/2**s), interpolation='nearest')
			plt.show()

		error = [np.zeros((batch, base_image_side/2**s, base_image_side/2**s)) for s in range(scales)]

		for s in range(scales):
			error[s] = I[s].reshape((batch, base_image_side/2**s, base_image_side/2**s)) - R[s]

		print "average L1 norm: " + str(np.sum(np.abs(A))/batch)

		if mod(t,50) == 0:
			display(t, Phi, base_image_side, save=save)
			print "Error norm across level 0: " + str(np.sum(error[0]**2))
			print "Error norm across level 1: " + str(np.sum(error[1]**2))
			print "Error norm across level 2: " + str(np.sum(error[2]**2))

		dPhi = [np.zeros((overcomplete*((base_image_side/2**s)-patch_side+1)**2, patch_dim)) for s in range(scales)]

		# Update basis function
		for s in range(scales):
			image_side = base_image_side/2**s
			neurons = int(image_side-(patch_side-1))**2

			if image_side == patch_side:
				dPhi[s] = np.dot(error[s].reshape(image_side*image_side, batch), A.T[:,mappings[s]]).T

			else:
				"""
				for i in range(neurons):
					loc = i % num_neurons_tile
					j = floor(loc/(image_side-(patch_side-1))) % (image_side-(patch_side-1)) 	# Current row
					k = loc % (image_side-(patch_side-1))		# Current col

					dPhi[:,i,s] = dot(error[s][j:j+patch_side,k:k+patch_side].reshape((patch_dim, batch)),A[i,:].T)
			# Fix vectorized learning rule
			# dPhi[:,i,:] = dot(error[j:j+patch_side,k:k+patch_side,:,:].reshape((patch_dim*scales,batch)),A[i,:].T).reshape((patch_dim,scales))
				"""
				#a = patchify(error[s], (patch_side, patch_side))
				#print a.shape
				for o in range(overcomplete):
					patches = patchify(error[s], (patch_side, patch_side)).reshape(batch, neurons, patch_dim)
					dPhi[s][o*neurons:(o+1)*neurons,:] = np.sum(patches*A.T[:,mappings[s][o*neurons:(o+1)*neurons]].reshape(batch, neurons, 1),axis=0)

		value = 0
		for s in range(scales):
			value = np.sum(error[s]**2) + value
		value = value + lambdav*np.sum(np.abs(A))
		print "Before value: " + str(value)

		newPhi = [np.zeros((((base_image_side/2**s)-patch_side+1)**2, patch_dim)) for s in range(scales)]
		newValue = float("inf")

		newValue = 0
		for s in range(scales):
			newPhi[s] = Phi[s] + (alpha**(scales-s-1)/float(batch)) * dPhi[s]
			newPhi[s] = newPhi[s]/np.sqrt(np.sum(newPhi[s]**2, axis=1))[:,None]

		A = sparsify(I, newPhi, mappings, lambdav)
		R = reconstruct(newPhi, A, mappings, base_image_dim)
		error = [np.zeros((batch, base_image_side/2**s, base_image_side/2**s)) for s in range(scales)]

		for s in range(scales):
			error[s] = I[s].reshape((batch, base_image_side/2**s, base_image_side/2**s)) - R[s]
			newValue = np.sum(error[s]**2) + newValue

		newValue = newValue + lambdav*np.sum(np.abs(A))
		alpha = max(0.25,beta*alpha)
		print "New value:" + str(newValue)
		print "alpha: " + str(alpha)

		for s in range(scales):
			Phi[s] = newPhi[s]

	return Phi

def reconstruct(Phi, A, mappings, base_image_dim, plot=False):
	# Phi: Basis functions (pixels, scales, image)
	# A: Coefficients basis function index. Each column separate image
	scales = len(Phi)
	(null, patch_dim) = Phi[0].shape
	(null, batch) = A.shape
	base_image_side = math.sqrt(base_image_dim)
	patch_side = math.sqrt(patch_dim)
	base_neurons = (base_image_side-(patch_side-1))**2
	overcomplete = Phi[0].shape[0]/base_neurons

	R = [np.zeros((batch, base_image_side/2**s, base_image_side/2**s)) for s in range(scales)]

	for s in range(scales):
		image_side = base_image_side/2**s
		neurons = (image_side-(patch_side-1))**2

		if image_side == patch_side:
			R[s] = np.dot(Phi[s].T, A[mappings[s],:]).reshape(batch, image_side, image_side)

		else:
			for i in range(int(overcomplete*neurons)):
				loc = i % neurons
				j = floor(loc/(image_side-(patch_side-1))) % (image_side-(patch_side-1)) # Current row
				k = loc % (image_side-(patch_side-1))		# Current col

				R[s][:,j:j+patch_side,k:k+patch_side] += \
					+ np.outer(A[mappings[s][i],:], Phi[s][i,:]).reshape(batch, patch_side,patch_side)
				#R[s][b,j:j+patch_side,k:k+patch_side] += \
				#	+ (A[mappings[s][i],b]*Phi[s][i,:]).reshape(patch_side,patch_side)
	if plot == True:
		#fig, ax = plt.subplots(scales)
		for s in range(scales):
			plt.imsave('test' + str(s),R[s][0,:,:])

	return R

def sparsify(I, Phi, mappings, lambdav, iterations=150, eta=0.1):
	"""Compute coefficients for reconstruction"""

	def g(u,theta, thresh_type='soft'):
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

	scales = len(I)
	(batch, base_image_dim) = I[0].shape
	base_image_side = math.sqrt(base_image_dim)
	scales = len(Phi)
	(null, patch_dim) = Phi[0].shape
	patch_side = math.sqrt(patch_dim)
	base_neurons = (base_image_side-(patch_side-1))**2
	overcomplete = Phi[0].shape[0]/base_neurons

	# Gamma = padded Phi
	Gamma = [np.zeros((overcomplete*(base_image_side/2**s-(patch_side-1))**2, base_image_side/2**s, base_image_side/2**s)) for s in range(scales)]

	for s in range(scales):
		image_side = base_image_side/2**s
		neurons = int(image_side-(patch_side-1))**2

		if image_side == patch_side:
			Gamma[s] = Phi[s].reshape(overcomplete*neurons, image_side, image_side)

		else:
			for i in range(int(overcomplete*neurons)):
				loc = i % neurons
				j = floor(loc/(image_side-(patch_side-1))) % math.sqrt(neurons) # Current row
				k = loc % (image_side-(patch_side-1))		# Current col
				Gamma[s][i,j:j+patch_side,k:k+patch_side] = Phi[s][i,:].reshape(patch_side, patch_side)

	b = np.zeros((overcomplete*base_neurons, batch))
	#G = sps.lil_matrix((neurons, neurons)) # sparse matrix
	G = np.zeros((overcomplete*base_neurons, overcomplete*base_neurons))

	# Compute b & G at base level
	Gamma[0] = Gamma[0].reshape((overcomplete*base_neurons, base_image_dim))
	b = b + np.dot(Gamma[0], I[0].T)
	G = G + np.dot(Gamma[0], Gamma[0].T)-np.eye(overcomplete*base_neurons)

	for s in range(1,scales):
		neurons = int(base_image_side/2**s-(patch_side-1))**2
		Gamma[s] = Gamma[s].reshape((overcomplete*neurons, base_image_dim/(2*2)**s))
		b_s = np.dot(Gamma[s],I[s].T)
		G_s = np.dot(Gamma[s], Gamma[s].T)-np.eye(overcomplete*neurons)
				# Vector-dot product of basis function with input
		b[mappings[s],:] = (b[mappings[s],:] + b_s)/2.0
		#print "sending: " + str(i*sqrt(neurons)+j) + " to " + str(ratio*(i*sqrt(base_neurons+j)))
		for o in range(int(overcomplete)):
			for i in range(int(sqrt(neurons))):
				for j in range(int(sqrt(neurons))):
					G[mappings[s][o*neurons+i],mappings[s][o*neurons+j]] = (G[mappings[s][o*neurons+i],mappings[s][o*neurons+j]] + G_s[o*neurons+i,o*neurons+j])/2.0

		# Pairwise dot product of each basis function
		# Want to normalize that part of matrix which has multiple additions
		#print G

	u = np.zeros((overcomplete*base_neurons,batch))

	l = 0.5 * np.max(np.abs(b), axis=0)
	a = g(u,l, 'soft')
	olda = a

	t = 0
	while (t < iterations+1) or (np.sqrt(np.sum((olda-a)**2)) > 10e-3):
		olda = a
		u = eta * (b-np.dot(G,a)) + (1-eta) * u
		a = g(u,l, 'soft')
		l = 0.98 * l
		l[l < lambdav] = lambdav

		t += 1

	print np.sum((a-olda)**2)

	return a

def display(t, Phi, base_image_side, save=False, overcomplete=1):
	scales = len(Phi)
	(base_neurons, patch_dim) = Phi[0].shape
	patch_side = sqrt(patch_dim)
	print "Iteration " + str(t)

	for s in range(scales):
		neurons = Phi[s].shape[0]
		image = -1*np.ones((patch_side*np.sqrt(neurons)+sqrt(neurons)+1,patch_side*sqrt(neurons)+sqrt(neurons)+1))
		for i in range(int(sqrt(neurons))):
			for j in range(int(sqrt(neurons))):
				temp = np.reshape(Phi[s][i*sqrt(neurons)+j,:],(patch_side,patch_side))-np.min(Phi[s][i*int(sqrt(neurons))+j,:])
				temp = temp/np.max(np.abs(Phi[s][i*sqrt(neurons)+j,:]))
				temp = 2*(temp-0.5)
				image[i*patch_side+i+1:i*patch_side+patch_side+i+1,j*patch_side+j+1:j*patch_side+patch_side+j+1] = temp

		plt.imsave('./figures/level' + str(s) + '-t' + str(t), image, cmap=cm.Greys_r)

		if save == True:
			np.save('./dictionaries/oc' + str(overcomplete) + '-s' + str(scales) +
				'-t' + str(t) + '-' + str(datetime.datetime.now()), Phi)

def patchify(img, patch_shape):
		# Images in (batch, side, side) format
		img = np.ascontiguousarray(img)  # won't make a copy if not needed
		batch, X, Y = img.shape
		x, y = patch_shape
		shape = (batch, (X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
		# The right strides can be thought by:
		# 1) Thinking of `img` as a chunk of memory in C order
		# 2) Asking how many items through that chunk of memory are needed when indices
		#    i,j,k,l are incremented by one
		strides = img.itemsize*np.array([X*Y, Y, 1, Y, 1])
		return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
