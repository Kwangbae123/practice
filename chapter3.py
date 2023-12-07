import chapter2
import numpy as np
import matplotlib.pylab as plt

def step_function_test(x):
    y = x>0  # numpy 배열 형태로 들어오면 출력은 bool값으로 반환된다.
    return y.astype(np.int) # bool값을 int 형태로 반환해서 출력(True : 1 / False : 0)

def step_function(x): #step_function 구현
    return np.array(x > 0, dtype=np.int)

# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1,1)
# plt.show()

def sigmoid(x): #sigmoid 함수 구현
    return 1 / (1 + np.exp(-x))

# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1,1)
# plt.show()

def relu(x): #ReLU 함수 구현
    return np.maximum(0, x) #maximun 두 입력중 큰값을 선택해 반환

# # 다차원 배열
# A = np.array([1, 2, 3, 4]) # 1차원 배열
# print(np.ndim(A)) # 배열의 차원수 반환
# print(A.shape) # 배열의 형상을 튜플로 반환한다.
# print(A.shape[0]) # shape[0] = 헹의 갯수 / shape[1] = 열의 갯수
#
# B = np.array([[1, 2],[3, 4],[5, 6]])
# print(B)
# print(np.ndim(B))
# print(B.shape)
#
# # 행렬의 곱
# A = np.array([[1, 2],[3, 4]])
# B = np.array([[5, 6],[7, 8]])
# np.dot(A,B) #행렬의 곱 연산
#
# # 신경망에서의 행렬곱
# X = np.array([1, 2])
# W = np.array([[1, 3, 5], [2, 4, 6]])
# Y = np.dot(X, W)
# print(Y)

# 4층 신경망에 대한 구현 및 정리
def init_network1():
    network = {} # 딕셔너리 초기화 각각 weight, bias를 저장한다.
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x): #forward propagation을 진행한다.
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)+ b3
    y = a3
    return y

# network = init_network1()
# x = np.array([1.0, 0.5])
# y = forward(network,x)
# print(y)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))
# softmax 함수의 출력은 0~1이고 총합은 1이다 이로인해 softmax의 함수의 출력을 확률로 해석이 가능하다.
# 문제를 확률적으로 대응 할수 있게 된다.
