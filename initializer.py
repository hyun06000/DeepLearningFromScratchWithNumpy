import numpy as np

class INIT:
    def __init__(self):
        pass
    '''
    FCN
    '''
    def initializer(self,size):
        self.W = np.random.randn(*size)
        self.DCDW = np.zeros_like(self.W)
        self.B = np.random.randn(size[-1])
        self.DCDB = np.zeros_like(self.B)
    
    def Xavier_initializer(self,size):
        self.W = np.random.randn(*size)/np.sqrt(size[1])
        self.DCDW = np.zeros_like(self.W)
        self.B = np.random.randn(size[-1])
        self.DCDB = np.zeros_like(self.B)
        
    def ones_initializer(self,size):
        self.W = np.ones(*size)
        self.DCDW = np.zeros_like(self.W)
        self.B = np.ones(size[-1])
        self.DCDB = np.zeros_like(self.B)
    '''
    CNN
    '''
    def conv_initializer(self,size):
        self.W = np.random.randn(size)
        self.DCDW = np.zeros_like(self.W)
        self.B = np.random.randn(size[-1])
        self.DCDB = np.zeros_like(self.B)
    
    def Xavier_conv_initializer(self,size):
        self.W = np.random.randn(size)/np.sqrt(size[1]*size[2])#나중에 알아볼것
        self.DCDW = np.zeros_like(self.W)
        self.B = np.random.randn(size[-1])
        self.DCDB = np.zeros_like(self.B)
    
    def conv_ones_initializer(self,size):
        self.W = np.ones(size)
        self.DCDW = np.zeros_like(self.W)
        self.B = np.ones(size[-1])
        self.DCDB = np.zeros_like(self.B)
    '''
    SELF
    '''
    def self_initializer(self,w,b):
        self.W = w
        self.DCDW = np.zeros_like(self.W)
        self.B = b
        self.DCDB = np.zeros_like(self.B)