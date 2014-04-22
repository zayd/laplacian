"""Learn dictionaries and sweep over parameters"""

import scales
import laplacian_pyramid

# lambdav = [0.001, 0.01, 0.05, 0.1, 0.2]
#lambdav = [0.2, 0.1, 0.075, 0.05, 0.025]
lambdav = [0.2, 0.1, 0.075]
base_mask_radius = 2

G = laplacian_pyramid.generative(64, 9,  3, base_mask_radius=base_mask_radius)

for l in lambdav:
	label = 'server-l' + str(l).split('.', 1)[1] + '-b' + str(base_mask_radius)
	print "Learning dictionary for l = " + str(l) + " with label " + label
	scales.learn(G=G, iterations=1500, inf_iterations=250,
		base_image_dim=64**2, lambdav=l, patch_dim=9**2, scales=3, 
		alpha=[400, 400, 400], label=label, plot_every=50, decrease_every=200, 
		save=True)