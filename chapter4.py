import numpy as np
import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist
import matplotlib.pyplot as plt

#오차제곱합(sum of squares for error, SSE)
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

#교차 엔트로피 오차(cross entropy error, CEE)
def cross_entropy_error(y, t):
    # delta = 1e-7 #오버플로우 방지 -무한대 발생하지 않도록
    # return -np.sum(t * np.log(y +delta))
    if y.ndim == 1: # y가 1차원이라면
        t = t.reshape(1, t.size) # t는 정답 레이블 reshape함수로 형상을 바꿔준다.
        y = y.reshape(1, y.size) # y는 신경망의 출력

    if t.size == y.size: # 훈련데이터가 원-핫-인코딩 벡터라면 label의 인덱스로 변환
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size # 정답 label이 '2', '7' 숫자 일때
    # np.log(y[np.arange(batch_size), t] -> np.arange(batch_size)sms 0 ~ batch_size-1 까지 배열 생성 / np.arange(batch_size), t와 각각 대응하는 출력을 추출한다.
    # return -np.sum(t * np.log(y + 1e-7)) / batch_size # 이미지 한장당 평균의 교차 엔트로피 오차를 계산한다.

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# print(x_train.shape)
# print(x_test.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # np.random.choice 는 지정한 범위의 수 중에서 무작위로 원하는 개수만 꺼낸다.
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

def numerical_diff(f, x): # 수치미분 중심차분(중앙차분법)
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

#function_1의 그래프
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
# plt.show()

#편미분
def function_2(x):
    return np.sum(x**2)

# x0=3, x1=4 일때 x0에대한 편미분


# 편미분을 벡터로 정리한것을 기울기(gradient)라 한다.
# 기울기(gradient) 구현
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx] # f(x+h) 계산
        x[idx] = tmp_val +h
        fxh1 = f(x)

        x[idx] = tmp_val - h # f(x-h) 계산
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #값 복원
        it.iternext()

    return grad

# print(numerical_gradient(function_2, np.array([3.0, 4.0])))

# gradient descent(경사하강법)
# f = 최적화 하려는 함수, init_x = 초기값, lr = learning rate, step_num = 반복횟수
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x) # numerical_gradient를 이용해 함수의 기울기를 구한다.
        x -= lr*grad

from chapter3 import softmax

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 정규분포로 초기화 한다.
    def predict(self, x): # 예측을 수행하는 메서드
        return np.dot(x, self.W)
    def loss(self, x, t): # 손실함수의 값을 구하는 메서드
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

net = simpleNet()
# print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
# print(p)
# print(np.argmax(p)) # 최대값의 인덱스
t = np.array([0, 0, 1]) # 정답레이블
# print(net.loss(x, t))

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)