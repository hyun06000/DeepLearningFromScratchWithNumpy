import numpy as np
import sys
from copy import deepcopy as copy

from initializer import INIT
from activation import ACTI
from convolution_network import CNN
from get_delta import DELTA
from optimizer import OPT

class Neuron(INIT,
             ACTI,
             CNN,
             DELTA,
             OPT):
    def __init__(self,x):
        self.X = x
        self.LAST = False
        self.LAYER = 'None'
        
        #Momentum
        self.V = 0
        self.GAMMA = 0.9
        
        #Adam
        self.BETA1,self.BETA2 = 0.9, 0.999
        self.MT = 0
        self.VT = 0
        
        #Gravity
        self.VW = 0
        self.VB = 0
        self.G = -9.8
        self.MASS = 1
        self.DRAG = 0
        
        
    def next_batch(self,x):
        self.X = x
            
    def FCN(self):
        x = self.X
        w = self.W
        b = self.B
        z = np.dot(x,w)
        z = z+b
        self.Z = z
        self.LAYER = 'FCN'
        
    def label(self,y):
        self.Y = y
        self.LAST = True
    
    def cost(self,fn):
        self.COST_FN = fn
        p = self.PREDIC
#         print(p)
        if fn == 'softmax_cross_entropy':
            return np.sum(self.Y*(-1)*np.log(p+1e-10))/p.shape[0]
        elif fn == 'sigmoid_cross_entropy':
            return np.sum(self.Y*(-1)*np.log(p+1e-10)+(1-self.Y)*(-1)*np.log(1-p+1e-10))/p.size
        elif fn == 'L2norm':
            return np.sum((self.Y-p)**2)/p.size
   