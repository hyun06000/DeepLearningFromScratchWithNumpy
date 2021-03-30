import numpy as np

class DELTA:
    def __init__(self):
        pass
    def delta(self,*fore):

        if self.LAYER == 'FCN':
            '''
            FCN_delta
            '''
            fn = self.COST_FN
            p = self.PREDIC
            if self.LAST:
                if fn == 'softmax_cross_entropy':
                    delta = p-self.Y

                elif fn == 'sigmoid_cross_entropy':
                    error = -1*self.Y/p
                    delta = error*self.d_act

                elif fn == 'L2norm':
                    d_act = self.d_act
                    delta = error*self.d_act

            else :
                fore[0] = fore_delta
                fore[1] = fore_w

                fore_error = np.dot(fore_delta,fore_w.T)
                delta = fore_error*self.d_act

            self.DELTA = delta

            for b in range(len(delta)):
                self.DCDW += np.outer(self.X[b,:],delta[b,:])/len(delta)
            self.DCDB = np.sum(delta,axis=0)

        elif self.LAYER == 'convolution':

            '''
            conv_back_prop
            '''

            x = self.X
            d_act = self.d_act
            fltrs = self.W
            strd = self.STRIDE
            fn = self.COST_FN
            '''

            !!!warning!!!
            you need convolution_transpose function in same class.

            ###args

            * fore_delta, fore_w, fore_strd = fore

            fore_delta : a tensor of delta from (l+1)th layer
                        when your local layer is (l)th layer.
                        it must be same shape as output.
                        [BATCH, DELTA_H, DELTA_W, OUT_CANNEL]

            fore_w : a convolution filter from (l+1)th layer.
                        [IN_CHANNEL, FILTER_H, FILTER_W, OUT_CHANNEL]

            fore_strd : a tuple or list of stride in (l+1)th layer.
                        [H STRIDE, WSTRIDE]

            ###self.args

            d_act : a ndarray which is derivative of activation function 

            x : input of local layer
                [BATCH, HEIGHT, WIDTH, IN_CHANNEL]

            fltrs : a convolution filter tensor of local layer.
                        it cna be different shape as fore_w
                        [IN_CHANNEL, FILTER_H, FILTER_W, OUT_CHANNEL]

            strd : a tuple or list of stride in local layer.
                        [H STRIDE, WSTRIDE]

            self.LAST:

            fn = self.COST_FN:

            ###returns

            z : a partial drivative of convolution filter.
                you need this for update local convolution filter.
                so it must be same shape as local convolution filter.
                [IN CHANNEL, FILTER_H, FILTER_W, OUT_CHANNEL]

            '''
            if self.LAST:
                if fn == 'softmax_cross_entropy':
                    delta = p-self.Y

                elif fn == 'sigmoid_cross_entropy':
                    error = -1*self.Y/p
                    d_act = self.d_act(self.Z)
                    delta = error*d_act

                elif fn == 'L2norm':
                    d_act = self.d_act(self.Z)
                    error = -2*(self.Y-p)
                    delta = error*d_act
            else :
                fore_delta, fore_w, fore_strd = fore

                error =  convolution(fore_delta,fore_w,fore_strd)
                delta = error*d_act

            SH,SW = strd[0],strd[1]
            (INCH,FH,FW,OUTCH) = filters.shape
            (B,DH,DW,OUTCH) = delta.shape

            dcdw = np.zeros_like(fltrs)
            for b in range(B):
                for h in range(DH):
                    for w in range(DW):
                        for outch in range(OUTCH):
                            for inch in range(INCH):
                                I = inputs[b,
                                           h*SH:h*SH+FH,
                                           w*SW:w*SW+FW,
                                           inch]

                                D = delta[b,h,w,outch]
                                dcdw[inch,:,:,outch] += I*D
            self.DCDW = dcdw
            self.DCDB  = np.sum(delta,axis=(0,1,2),keepdims=-1)
        elif self.LAYER == 'convolution_transpose':
            '''
            conv_transpose_back_prop
            '''

            x = self.X
            d_act = self.d_act
            fltrs = self.W
            strd = self.STRIDE
            fn = self.COST_FN
            '''

            !!!warning!!!
            you need convolution function in same class.

            ###args

            * fore_delta, fore_w, fore_strd = fore

            fore_delta : a ndarray of delta from (l+1)th layer
                        when your local layer is (l)th layer.
                        it must be same shape as output.
                        [BATCH, DELTA_H, DELTA_W, OUT_CANNEL]

            fore_w : a ndarray of convolution filter from (l+1)th layer.
                        [IN_CHANNEL, FILTER_H, FILTER_W, OUT_CHANNEL]

            fore_strd : a tuple or list of stride in (l+1)th layer.
                        [H STRIDE, W STRIDE]

            ###self.args

            d_act = self.d_act : a ndarray which is derivative of activation function 

            x = self.X : input of local layer
                [BATCH, HEIGHT, WIDTH, IN_CHANNEL]

            fltrs = self.W : a convolution filter tensor of local layer.
                        it cna be different shape as fore_w
                        [IN_CHANNEL, FILTER_H, FILTER_W, OUT_CHANNEL]

            strd = self.STRIDE : a tuple or list of stride in local layer.
                        [H STRIDE, WSTRIDE]

            self.LAST :

            fn = self.COST_FN

            ###returns

            z : a partial drivative of convolution_transpose filter.
                you need this for update local convolution_transpose filter.
                so it must be same shape as local convolution_transpose filter.
                [IN CHANNEL, FILTER_H, FILTER_W, OUT_CHANNEL]

            '''



            if self.LAST:
                if fn == 'softmax_cross_entropy':
                    delta = p-self.Y

                elif fn == 'sigmoid_cross_entropy':
                    error = -1*self.Y/p
                    d_act = self.d_act(self.Z)
                    delta = error*d_act

                elif fn == 'L2norm':
                    d_act = self.d_act(self.Z)
                    error = -2*(self.Y-p)
                    delta = error*d_act
            else :
                fore_delta,fore_w,fore_strd = fore
                error =  convolution(fore_delta,fore_w,fore_strd)
                delta = error*d_act


            error = convolution(fore_delta,fore_w,fore_strd)
            delta = error * d_act

            dcdw = np.zeros_like(fltrs)
            for b in range(B):
                for outch in range(outCH):
                    for inch in range(inCH):
                        for inh in range(inH):
                            for inw in range(inW):

                                I = x[b,inh,inw,inch]

                                D = delta[b,
                                         inh*sH : inh*sH+fH,
                                         inw*sW : inw*sW+fW,
                                         outch]
                                dcdw[inch,:,:,outch] += I*D
            self.DCDW = dcdw
            self.DCDB = np.sum(delta,axis=(0,1,2),keepdims=-1)
