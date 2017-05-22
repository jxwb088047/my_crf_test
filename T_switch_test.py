# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 14:55:14 2017

@author: B907-LGH
"""

import theano.tensor as T
import theano
a,b=T.dmatrices('a','b')
x,y=T.dvectors('x','y')
z=T.switch(T.lt(a,b),x,y)


test_switch=theano.function([a,b,x,y],z)


import numpy as np
X,Y=np.random.randn(4),np.random.randn(4)

A,B=np.random.randn(3,4),np.random.randn(3,4)

print 'X:'
print X
print 

print 'Y:'
print Y
print 

print 'A:'
print A
print 

print 'B:'
print B
print 

print 'result:'
print test_switch(A,B,X,Y)
print 

print A<B


