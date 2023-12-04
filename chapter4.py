import numpy as np
import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist
import matplotlib.pylab as plt

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

    for idx in range(x.size):
        tmp_val = x[idx] # f(x+h) 계산
        x[idx] = tmp_val +h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #값 복원

    return grad

# print(numerical_gradient(function_2, np.array([3.0, 4.0])))
