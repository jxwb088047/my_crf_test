# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:58:32 2017

@author: B907-LGH
"""

#pe
import numpy as np
def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
        print i, encoding[i-1,:]
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them 
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

position_encoding(3,4)
print 

def position_encoding2(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    
    for j in np.arange(sentence_size):
        encoding[:,j]=(1-(j+1)/sentence_size)-((np.arange(embedding_size)+1)/embedding_size)*(1-2*(j+1)/sentence_size)
        print j
        print encoding[:,j]
    return encoding

print position_encoding2(3,4)        
print


def position_encoding3(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    
    for j in xrange(sentence_size):
        for i in xrange(embedding_size):
            encoding[i][j]=(1-(j+1)/sentence_size)-((i+1)/embedding_size)*(1-2*(j+1)/sentence_size)
        print j
        print encoding[:,j]

    print encoding

position_encoding3(3,4)
print