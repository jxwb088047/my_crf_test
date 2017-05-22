# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 18:46:23 2017

@author: B907-LGH
"""

import theano
import theano.tensor as T
import numpy as np

# define tensor variables
X = T.vector("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")
U = T.matrix("U")
Y = T.matrix("Y")
V = T.matrix("V")
P = T.matrix("P")

results, updates = theano.scan(lambda y, p, x_tm1: T.tanh(T.dot(x_tm1, W) + T.dot(y, U) + T.dot(p, V)),
          sequences=[Y, P[::-1]], outputs_info=[X])
compute_seq = theano.function(inputs=[X, W, Y, U, P, V], outputs=results)

# test values
x = np.zeros((2), dtype=theano.config.floatX)
x[1] = 1
print 'x:',x

w = np.ones((2, 2), dtype=theano.config.floatX)
print 'w:',w

y = np.ones((5, 2), dtype=theano.config.floatX)
y[0, :] = -3
print 'y:',y

u = np.ones((2, 2), dtype=theano.config.floatX)
print 'u:',u

p = np.ones((5, 2), dtype=theano.config.floatX)
p[0, :] = 3

print 'p:',p

v = np.ones((2, 2), dtype=theano.config.floatX)
print 'v:',v

print 
print(compute_seq(x, w, y, u, p, v))

# comparison with numpy
x_res = np.zeros((5, 2), dtype=theano.config.floatX)
x_res[0] = x.dot(w) + y[0].dot(u) + p[4].dot(v)
#np.tanh(#)
for i in range(1, 5):
    x_res[i] = np.tanh(x_res[i - 1].dot(w) + y[i].dot(u) + p[4-i].dot(v))

print(x_res)


#目前的理解为：
#scan中的【X】为初始化，感觉实际是X【-1】位置