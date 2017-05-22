# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 09:05:58 2017

@author: B907-LGH
"""

import theano.tensor as T
import theano

def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def forward(observations, transitions, viterbi=False,
            return_alpha=False, return_best_sequence=False):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities must be given in the log space.
    Compute alpha, matrix of size (n_steps, n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
    """
    assert not return_best_sequence or (viterbi and not return_alpha)

    def recurrence(obs, previous, transitions):
        previous = previous.dimshuffle(0, 'x')
        obs = obs.dimshuffle('x', 0)
        if viterbi:
            scores = previous + obs + transitions
            out = scores.max(axis=0)
            if return_best_sequence:
                out2 = scores.argmax(axis=0)
                return out, out2
            else:
                return out
        else:
            return log_sum_exp(previous + obs + transitions, axis=0)

    initial = observations[0]
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=(initial, None) if return_best_sequence else initial,
        sequences=[observations[1:]],
        non_sequences=transitions
    )

    if return_alpha:
        return alpha
    elif return_best_sequence:
        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(alpha[0][-1]), 'int32'),
            sequences=T.cast(alpha[1][::-1], 'int32')
        )
        sequence = T.concatenate([sequence[::-1], [T.argmax(alpha[0][-1])]])
        return alpha[0],alpha[1],sequence
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)
            
            
if __name__=='__main__':
    import numpy as np

    np.random.seed(1234)
    trans=np.random.rand(4,4).astype(np.float32)
    
    #trans.dtype=np.float32
    
    print 'trans:'
    print trans
    print 'dtype:',trans.dtype
    print         
    
    obvs=np.random.randint(0,10,size=(3,4)).astype(np.float32)
   # obvs.shape=(3,4)
    print 'obvs:'
    print obvs    
    print 'dtype:',obvs.dtype
    print 
    

    
    observations=T.matrix('observations')
    transitions=T.matrix('transitions')
    
    func=theano.function([observations,transitions],forward(observations=observations,transitions=transitions,viterbi=True,
            return_alpha=False, return_best_sequence=True))
    alpha0,alpha1,sequence=func(obvs,trans)
    
    print 'alpha0:'
    print alpha0
    print 
    
    print 'alpha1:'
    print alpha1
    print 
    
    
    print 'sequences:'
    print sequence
    print 
    
    
    
    