import sys, os
sys.path.append(os.pardir)

import numpy as np
from util.function import softmax
from util.function import cross_entropy_error

# 곱셈계층의 역전파 구현
class MulLayer:
    def __init__(self):
        self.x = None # x 초기화
        self.y = None # y 초기화

    def forward(self, x, y):
        self.x = x # x를 받는다.
        self.y = y # y를 받는다.
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y # x와 y를 바꾼다
        dy = dout * self.x
        return dx, dy

# 덧셈 계층
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

# ReLU 함수의 순전파 역전파 구현
class ReLU:
    def __init__(self):
        # mask는 True/False 로 구성된 numpy 배열
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # x <= 0 이면 False / x > 0 이면 True
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0 # 순전파에서 만들었던걸 이용해 False 일때 0이 역전파 되도록 설계한다.
        dx = dout
        return dx


# sigmoid 함수의 순전파 역전파 구현
class Sigmoid:
    def __init__(self):
        self.ot = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# Affine 계층 구현
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


# Softmax-with-Loss
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실
        self.y = None # softmax의 출력
        self.t = None # 정답 label(one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx