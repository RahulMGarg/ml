import theano
import theano.tensor as T
import cPickle as pickle
import numpy as np

def load_if_saved(x):
    return x

x = T.matrix()
y = T.scalar()

out = y * x

t_fn = theano.function([x, y], out)
X = np.array([[1, 2], [3, 4]])
c = 2.0
print t_fn(X, c)
