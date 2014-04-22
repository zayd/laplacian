""" LCA Implementaion for multiscale Laplacian Pyramid """

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