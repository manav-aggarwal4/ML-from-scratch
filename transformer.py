import numpy as np

class DecoderBlock:
    def __init__(self, d_model, d_ff, num_heads):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.wK = np.random.randn(d_model, d_model)
        self.wQ = np.random.randn(d_model, d_model)
        self.wV = np.random.randn(d_model, d_model)
        self.wO = np.random.randn(d_model, d_model)

        self.FF1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.FF2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)

        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def softMax(self, x):
        """
        Subtract maximum value in x so exponential's don't overflow
        do e^ some x / sum of e^ every other x
        """
        maximum = np.max(x, axis=-1, keepdims=True)
        exponential = np.exp(x - maximum)
        return exponential / np.sum(exponential, axis=-1, keepdims=True)
    
    def LayerNorm(self, x, gamma, beta): 
        """
        Find mean, variance of x and normalize -> mean = 0, variance = 1
        gamma and beta are learned scales

        """
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        xNorm = (x - mean) / np.sqrt(var)

        return gamma * xNorm + beta
    
    def MHA(self, x, mask):
        B, T, _ = x.shape
        K = x @ self.wK
        Q = x @ self.wQ
        V = x @ self.wV
        d_head = self.d_model // self.num_heads
        K = K.reshape(B, T, self.num_heads, d_head)
        Q = Q.reshape(B, T, self.num_heads, d_head)
        V = V.reshape(B, T, self.num_heads, d_head)
        K = K.transpose(0, 2, 1, 3)
        Q = Q.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        
        QKT = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_model)

        if mask is not None:
            if mask.ndim == 2:
                maskReshaped = mask.reshape(1, 1, T, T)
            else:
                maskReshaped = mask
            QKT += maskReshaped
        attn_weights = self.softMax(QKT)

        head_out = attn_weights @ V

        head_out = head_out.transpose(0, 2, 1, 3)  

        out = head_out.reshape(B, T, self.d_model)

        out = out @ self.wO

        return out


    def feedForward(self, x):

    