import numpy as np

class OPT:
    def __init__(self):
        pass
        
    def update(self,learning_rate):
        
        self.W -= learning_rate*self.DCDW
        self.B -= learning_rate*self.DCDB
        
    def Momentum(self,learning_rate):
        
        self.V = self.GAMMA*self.V+self.DCDW
        self.W -= self.V
        self.B -= learning_rate*self.DCDB
    
    def Adam(self,learning_rate):
        
        beta1,beta2 = self.BETA1,self.BETA2
        grad = self.DCDW
        mt = self.MT
        vt = self.VT
        
        mt = beta1*mt+(1-beta1)*grad
        vt = beta2*vt+(1-beta2)*(grad**2)
        self.W -= learning_rate*(mt/(1-beta1))/(np.sqrt((vt/(1-beta2)+1e-8)))
        
        self.MT = mt
        self.VT = vt
        
    def Gravity(self,learning_rate):
        def sine(grad):
            theta = np.arctan(grad)
            return np.sin(theta)
        def cosine(grad):
            theta = np.arctan(grad)
            return np.cos(theta)
        
        self.VW += self.G*sine(self.DCDW)*cosine(self.DCDW)*learning_rate - (self.DRAG/self.MASS)*self.VW*learning_rate
        self.W += self.VW*learning_rate
        
        self.VB += self.G*sine(self.DCDB)*cosine(self.DCDB)*learning_rate - (self.DRAG/self.MASS)*self.VB*learning_rate
        self.B += self.VB*learning_rate
        