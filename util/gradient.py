import numpy as np

# 수치미분 중심차분(중앙차분법)
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)

# 편미분을 벡터로 정리한것을 기울기(gradient)라 한다.
# 기울기(gradient) 구현
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx] # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h # f(x-h) 계산
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #값 복원
        it.iternext()
    return grad

# gradient descent(경사하강법)
# f = 최적화 하려는 함수, init_x = 초기값, lr = learning rate, step_num = 반복횟수
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x) # numerical_gradient를 이용해 함수의 기울기를 구한다.
        x -= lr*grad