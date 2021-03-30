import numpy as np

class ACTI:
    def __init__(self):
        pass
    
    def relu(self):
        self.PREDIC = np.fmax(np.zeros_like(self.Z),self.Z) 
        
    def d_relu(self):
        self.d_act = np.fmin(np.ones_like(self.Z),self.Z) 
        
    def softmax(self):
        e_x = np.exp(self.Z-np.amax(self.Z,axis=-1,keepdims=-1))
        self.PREDIC = np.sum(e_x,axis=-1,keepdims=-1)
        
    
    
    def activation(self,act):
        x = self.Z
        if act == 'sigmoid':
            self.PREDIC = 1./(1.+np.exp(-x)) 
            self.d_act = self.PREDIC*(1-self.PREDIC)
            
        elif act == 'relu':
            self.PREDIC = np.fmax(np.zeros_like(x),x) 
            self.d_act = np.fmin(np.ones_like(self.PREDIC),self.PREDIC) 
            
        elif act == 'softmax':
            e_x = np.exp(x-np.amax(x,axis=-1,keepdims=-1))
            self.PREDIC = e_x/np.sum(e_x,axis=-1,keepdims=-1)
            #d_act : we will use derivative of softmax-cross-entropy
        
        
        