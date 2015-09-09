import time
import theano
import theano.tensor as T
import cPickle as pickle
import numpy as np

def cached_function(name, *args, **kwargs):
    '''
    Loads a theano function from disk, else compiles and pickles it.
    '''
    filename = '%s.pkl' % name
    try:
        fn = pickle.load(open(filename))
    except (OSError, IOError) as e:
        fn = theano.function(*args, **kwargs)
        pickle.dump(fn, open(filename, 'w'))
        fn.filename = name
    return fn

if __name__=='__main__':
    x = T.matrix()
    y = T.scalar()

    out = y * x

    t = time.time()
    t_fn = cached_function('t_fn', [x, y], out)
    print time.time() - t
    X = np.array([[1, 2], [3, 4]])
    c = 2.0
    print t_fn(X, c)
