import numpy as np

class CNN:
    def __init__():
        pass
    
    def convolution(self,
                    strd):
        
        x = self.X
        ftr = self.W
        self.STRIDE = strd
        
        
        '''
        ### args
        strd : filter stride number. 
                [H STRIDE, W STRIDE]

        ### self.args

        x : input of convolution.
                [BATCH, HEIGHT, WIDTH, CHANNEL]
        ftr : filter for convolution operation.
                [IN CHANNEL, FILTER H, FILTER W, OUT CHANNEL]
        

        ### retruns

        z : result of convolutional neural network operation. 
            the input size and the filter size is make new smaple size with stride.
            so we call it 'hopes'. then output size is
            [BATCH, H HOPES, W HOPES, OUT CHANNEL]
        '''

        (B,H,W,C) = x.shape
        (in_C,f_H,f_W,out_C) = ftr.shape
        (strd_H,strd_W) = strd

        H_hopes = int((H - f_H)/strd_H + 1)
        W_hopes = int((W - f_W)/strd_W + 1)

        z = np.zeros((B,H_hopes,W_hopes,out_C))

        for b in range(B):#배치에서 한장을 가져옴
            for out_ch in range(out_C):
                for h in range(H_hopes):#우선 가로로 움직이기 위해 행을 선택해줌
                    h_start = h*strd_H
                    for w in range(W_hopes):#이제 선택된 행 안에서 움직임
                        w_start = w*strd_W
                        for ch in range(C):
                            x_ = x[b,
                                   h_start:h_start+f_H,
                                   w_start:w_start+f_W,
                                   ch]
                            z[b,h,w,out_ch] += np.sum(x_*ftr[ch,:,:,out_ch])
        self.Z = z + self.B
        self.LAYER = 'CONVOLUTION'
        
        
    def convolution_transpose(self,
                              strd):

        x = self.X
        ftr = self.W
        self.STRIDE = strd
        
        '''

        ### args
        
        strd : filter stride number. 
                [H STRIDE, W STRIDE]

        ### slef.args
        
        x = self.X : input of convolution.
                [BATCH, HEIGHT, WIDTH, CHANNEL]
        ftr = self.W : filter for convolution operation.
                [IN CHANNEL, FILTER H, FILTER W, OUT CHANNEL]
        
        self.LAST :
        
        ### retruns

        z : result of convolutional neural network operation. 
            the input size and the filter size is make new smaple size with stride.
            so we call it 'hopes'. then output size is
            [BATCH, H HOPES, W HOPES, OUT CHANNEL]
        '''

        (B,H,W,C) = x.shape
        (in_C,f_H,f_W,out_C) = ftr.shape
        (strd_H,strd_W) = strd

        H_hopes = (H-1)*strd_H+f_H
        W_hopes = (W-1)*strd_W+f_W

        z = np.zeros((B,H_hopes,W_hopes,out_C))

        for b in range(B):#배치에서 한장을 가져옴
            for out_ch in range(out_C):
                for h in range(H):#우선 가로로 움직이기 위해 행을 선택해줌
                    for w in range(W):#이제 선택된 행 안에서 움직임
                        f_sum = 0
                        for ch in range(C):
                            x_ = x[b,h,w,ch]
                            f_sum += x_*ftr[ch,:,:,out_ch]
                        z[b,
                          h*strd_H:(h*strd_H)+f_H,
                          w*strd_W:(w*strd_W)+f_W,
                          out_ch] += f_sum
        self.Z = z + self.B
        self.LAYER = 'CONVOLUTION_TRANSPOSE'
        