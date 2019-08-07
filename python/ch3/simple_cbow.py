import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss, Embedding

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # self.in_layer0 = MatMul(W_in)
        # self.in_layer1 = MatMul(W_in)
        # self.out_layer = MatMul(W_out)
        self.in_layer0 = Embedding(W_in)
        self.in_layer1 = Embedding(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        
        h0 = self.in_layer0.forward(np.argmax(contexts[:, 0], axis=1))
        h1 = self.in_layer1.forward(np.argmax(contexts[:, 1], axis=1))
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        print(score.shape, target.shape)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None

if __name__ == '__main__':
    cbow = SimpleCBOW(5, 10)
