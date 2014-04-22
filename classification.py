# Classification task on CIFAR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import laplacian

def unpickle(file):
    # Open CIFAR files
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def classify(Phi,data='data_batch_1'):

  laplacian.laplacian(images='cifar', num_images=)
  
  # Linear Classifier in Pixel Domain
  cifar = unpickle('./data/cifar-10/' + data)
  X = cifar['data']
  y = cifar['labels']
  #pixel = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)

  # Load laplacian pyramid  
  Phi = np.load('./dictionaries/oc1-s3-t500.npy')
  X = laplacian.laplacian(images='cifar')
  laplacian.sparsify(X, Phi, lambdav=0.1)
  pyramid = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)

  error = np.sum(y==prediction)/np.shape(y)[0]

  return prediction