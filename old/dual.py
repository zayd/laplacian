# Solve ||X-BS||^2 s.t. ||B_i||^2 < c  using the lagrange dual

def dual(X, B, S):
	"""
	X: image colums vector, B: basis column vector, S: coefficient column vectors
	"""
	XSt = X.dot(S.T)
	SStL = np.inv(S.dot(S.T)+np.diag(L))
	
	dL  = 
	d2L = -2*(SStL.dot(XSt.T.dot(XSt)).dot(SStL)) * SStL