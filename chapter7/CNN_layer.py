# 합송곱 계층 구현하기 P.245
import sys,os
sys.path.append(os.pardir)
import numpy as np
from util.util import im2col, col2im

x1 = np.random.rand(1, 3, 7, 7) # (데이터수, 채널수, 높이, 너비)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7) # (데이터수, 채널수, 높이, 너비)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)

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


























