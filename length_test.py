# -*- coding: utf-8 -*-
"""
Created on Fri May 05 19:42:28 2017

@author: B907-LGH
"""



from keras.layers import  LSTM

import theano.tensor as T
import theano

def create_func(inputs,outputs=None):
    return theano.function(inputs=inputs,outputs=outputs if outputs else inputs)
    
import numpy as np
x_v=np.random.randn(6,4).astype(theano.config.floatX)
print 'x_v:'
print x_v
print
y_v=np.random.randn(2).astype(theano.config.floatX)
print 'y_v:'
print y_v
print

X=T.matrix('X')
y=T.vector('y')

new_X=T.set_subtensor(X[2:4,2:],y)

replace_part_func=create_func([X,y],new_X)
print 'Replace part function:'
print replace_part_func(x_v,y_v)
print

