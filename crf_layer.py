# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:05:14 2017

@author: B907-LGH
"""
from keras import backend as K

from keras.engine.topology import Layer
from keras import initializations

import theano
import theano.tensor as T
from theano.ifelse import  ifelse
from theano import shared

import numpy as np

def create_function(inputs,outputs=None):
    return theano.function(inputs=inputs,outputs=outputs if outputs else inputs)

class CRFLayer(Layer):
    
    def __init__(self,tagger_num,init='glorot_uniform',**kwargs):
        self.tagger_num=tagger_num
        self.init=initializations.get(init)
        super(CRFLayer,self).__init__(**kwargs)
    
    def build(self,input_shape):

         self.transitions=self.init((self.tagger_num+2, self.tagger_num+2),
                               name='{}_transitions'.format(self.name))
                               
         self.trainable_weights = [self.transitions]
         super(CRFLayer,self).build(input_shape)
        
    def call(self,x):
        return x
        
    def compute_output_shape(self,input_shape):
        return input_shape
        
    def forward(obvs,trans):
        
        pass
        
    def step_crf_loss(self,o, t):
        
        #向量情况下
        self.transitions=self.init((self.tagger_num+2, self.tagger_num+2),
                               name='{}_transitions'.format(self.name))
                               
        self.trainable_weights = [self.transitions]
        
        output=T.matrix('output')
        target=T.imatrix('target')

        maxLength=target.shape[0]

        index=T.argmax(target,axis=1)
        sums=T.sum(target,axis=1)

        
        padding=T.pow(2,self.tagger_num)-1
        new_target=T.switch(T.eq(sums,self.tagger_num*T.ones_like(sums)),padding*T.ones_like(index),index)

        L=ifelse(T.eq(T.max(new_target),padding),T.argmax(new_target),maxLength)
        L_func=create_function([target],[L])
        print 'Length:'
        print L_func(t)
        print
        
        no_padding_target=new_target[:L]
        
        target_func=create_function([target],[target,new_target,no_padding_target])
        t_1,n_t_1,n_p_t_1=target_func(t)
        
        
        print 'target:'
        print t_1
        print
        
        print'new_target:'
        print n_t_1
        print
        
        print 'no_padding_target:'
        print n_p_t_1
        print
        
        #shared 函数value必须是实值
        b_id=shared(value=np.array([self.tagger_num],dtype=np.int32))
        e_id=shared(value=np.array([self.tagger_num+1],dtype=np.int32))
        
        with_b_e_no_padding_target=T.concatenate([b_id,no_padding_target,e_id],axis=0)
        
        
        print 'L:'
        print L_func(t)[0]
        print 
        
        print 'output:'
        print o
        print o[np.arange(L_func(t)[0]),n_p_t_1].sum()
        print 
        
        real_score=output[T.arange(L),no_padding_target].sum()
        
        real_score_func=create_function([output,target],real_score)
        print 'real_score:'
        print real_score_func(o,t)
        print
        
        print 'Real score equal or not:'
        print np.allclose(real_score_func(o,t),o[np.arange(L_func(t)[0]),n_p_t_1].sum())
        print 

        s=real_score+ \
            self.transitions[with_b_e_no_padding_target[T.arange(L+1)],\
                             with_b_e_no_padding_target[T.arange(L+1)+1]].sum()
                             
        print 'transitions:'
        print self.transitions
        print 

        small = -100
        
        observations=T.concatenate([output,small*T.ones(shape=(maxLength,2))],axis=1)
        b_s=np.array([self.tagger_num*[small]+[0,small]],dtype=np.int32)
        e_s=np.array([self.tagger_num*[small]+[small,0]],dtype=np.int32)
        observations=T.concatenate([b_s,observations,e_s],axis=0)
        
        #cost=s-forword(observations,self.transitions)
        
        s_func=create_function([output,target],[s])
        print 'Output:',s_func(o,t)
        
        
        pass


#if __name__ == '__main__':
if __name__=='__main__':
    sequence_length=5
    tagger_num=5
    crf_layer=CRFLayer(tagger_num=tagger_num)
    o=np.random.randn(sequence_length,tagger_num).astype(np.float32)
    t=np.array([[0,1,0,0,0],
                [0,0,1,0,0],
                [1,0,0,0,0],
                [0,0,0,1,0],
                [1,1,1,1,1]]).astype(np.int32)
    print 'CRF Loss:'
    print crf_layer.step_crf_loss(o,t)
    
    print 
