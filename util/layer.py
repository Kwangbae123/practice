import sys, os
sys.path.append(os.pardir)

import numpy as np
from util.function import softmax
from util.function import cross_entropy_error
from util.util import im2col, col2im

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

#합성곱 계층 구현
class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape # FN : 필터갯수 / C : 채널수, FH : 필터 높이, FW : 필터 너비
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad) # 입력데이터를 im2col을 사용 해 필터를 전개한다.
        # 필터도 reshape을 통해 전개한다. / reshape -1은 다차원 배열의 원소수가 변황후에도 똑같게 설정
        # (10, 3, 5, 5) 다차원 배열에서 w의 원소수는 750개 reshape(10, -1)을 하면 (10, 75)인 배열로 만들어준다.
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        # transpose 함수를 사용해 배열의 순서를 변경한다.
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx


# 풀링 계층 구현
class Pooling:
    def __init__(self, pool_h, pool_w, stride = 1, pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 전개(1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 전개한 값에서 영역별 최댓값 추출한다.
        out = np.max(col, axis = 1)

        # 출력데이터 형상에 맞게 reshape 해준다.
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx
