import numpy as np
import sys
from copy import deepcopy as copy

class Neuron:
    def __init__(self,x):
        self.X = x
        self.V = 0
        self.LAST_LAYER = False
    
    def next_batch(self,x):
        self.X = x
            
    def initialize(self,size):
        self.W = np.random.randn(size[0],size[1])
        self.DCDW = np.zeros_like(self.W)
        self.B = np.random.randn(size[1])
        self.DCDB = np.zeros_like(self.B)
    
    def Xavier_initialize(self,size):
        self.W = np.random.randn(size[0],size[1])/np.sqrt(size[1])
        self.DCDW = np.zeros_like(self.W)
        self.B = np.random.randn(size[1])
        self.DCDB = np.zeros_like(self.B)
        
    def self_initializer(self,w,b):
        self.W = w
        self.DCDW = np.zeros_like(self.W)
        self.B = b
        self.DCDB = np.zeros_like(self.B)

    def FCN(self):
        x = self.X
        w = self.W
        b = self.B
        z = np.dot(x,w)
        z = z+b
        self.Z = z
        
    def relu(self,z):
        OUTPUT = np.fmax(np.zeros_like(z),z) 
        return OUTPUT
    
    def d_relu(self,z):
        OUTPUT = np.fmin(np.ones_like(z),z) 
        return OUTPUT
        
    def sigmoid(self,z):
        o = 1./(1.+np.exp(-z)) 
        return o
    
    def d_sigmoid(self,x):
        s = self.sigmoid
        return s(x)*(1-s(x))
    
    def softmax(self,x):
        e_x = np.exp(x-np.amax(x,axis=-1,keepdims=-1))
        e_x_s = np.sum(e_x,axis=-1,keepdims=-1)
        return e_x / e_x_s
    
    
    def activation(self,act):
        if act == 'relu':
            self.act = self.relu
            self.d_act = self.d_relu
            
        elif act == 'sigmoid':
            self.act = self.sigmoid
            self.d_act = self.d_sigmoid
        
        elif act == 'softmax':
            self.act = self.softmax
            #we will use derivative of softmax-cross-entropy
        self.PREDIC = self.act(self.Z)
        return self.PREDIC
        
    def label(self,y):
        self.Y = y
        self.LAST_LAYER = True
#         print('LABEL is commited')
    
    def cost(self,fn):
        self.cost_fn = fn
        p = self.PREDIC
#         print(p)
        if fn == 'softmax_cross_entropy':
            return -1*np.sum(self.Y*np.log(p+1e-10))
        elif fn == 'sigmoid_cross_entropy':
            return -1*np.sum(self.Y*np.log(p))
        elif fn == 'L2norm':
            return np.sum((self.Y-p)**2)
        
    def delta(self,*fore_val):
        fn = self.cost_fn
        p = self.PREDIC
        d_act = self.d_act(self.Z)
        
        if self.LAST_LAYER:
            if fn == 'softmax_cross_entropy':
                delta = p-self.Y

            elif fn == 'sigmoid_cross_entropy':
                error = -1*self.Y/p
                delta = error*d_act

            elif fn == 'L2norm':
                error = -2*(self.Y-p)
                delta = error*d_act
        else :
            fore_delta = fore_val[0]
            fore_w = fore_val[1]
            
            error = np.dot(fore_delta,fore_w.T)
            delta = error*d_act
        
        self.DELTA = delta
        for b in range(len(delta)):
            self.DCDW += np.outer(self.X[b,:],delta[b,:])/len(delta)
        self.DCDB = np.sum(delta,axis=0)
        
    def update(self,learning_rate):
        
        self.W -= learning_rate*self.DCDW
        self.B -= learning_rate*self.DCDB
        
    def momentum(self,learning_rate,gamma):
        
        self.V = gamma*self.V+self.DCDW
        self.W -= self.V
        self.B -= learning_rate*self.DCDB
        
        
        
        