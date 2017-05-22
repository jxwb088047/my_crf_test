# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 17:58:09 2017

@author: B907-LGH
"""

import theano
import theano.tensor as T
import numpy as np

# defining the tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v:T.tanh(
            T.dot(v, W) + b_sym
            )
            , sequences=X)
compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=results)

# test values
x = np.eye(3, dtype=theano.config.floatX)
w=np.arange(1,7).astype(dtype=theano.config.floatX)
w.shape=(3,2)
#w = np.ones((3, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
b[1] = 2

print 'x:',x
print 'w:',w
print 'b:',b

#注意点：
#这里是按【行】作为【step】

print(compute_elementwise(x, w, b))

# comparison with numpy
print(np.tanh(x.dot(w) + b))