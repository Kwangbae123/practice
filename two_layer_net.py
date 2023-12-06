# 2층 신경망 클래스 구현
import os, sys
import numpy as np
sys.path.append(os.pardir)
from chapter3 import *
from chapter4 import *
from chapter5 import *
from _collections import OrderedDict #순서가 있는 딕셔너리 생성 모듈

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01): # 입력층의 뉴련수,은닉층의 뉴런수, 출력층의 뉴런수 초기화
        # 가중치 초기화
        self.params ={} # 신경망의 매개변수들을 저장하는 딕셔너리 변수
        # W1 = 1번째 층의 가중치 , b1 = 1번째 층의 bias
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayers = SoftmaxWithLoss

    def predict(self, x): # 예측을 진행한다. x는 이미지 데이터
        # W1 ,W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']
        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # y = softmax(a2)
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x : 입력데이터 , t = 정답 label
    def loss(self, x, t): # 손실함수 계산 x = 이미지 데이터, t = 정답 label
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t): # 정확도를 구한다.
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t): # 가중치 매개변수의 기울기를 구한다.
        loss_W = lambda W: self.loss(x, t)

        grads = {} # 기울기를 보관하는 딕셔너리 변수
        # W1 = 1번째 층의 가중치의 기울기 , b1 = 1번째 층의 bias의 기울기
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        #역전파
        dout = 1
        dout = self.lastLayers.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #결과 저장
        grads = {} # 기울기를 보관하는 딕셔너리 변수
        # W1 = 1번째 층의 가중치의 기울기 , b1 = 1번째 층의 bias의 기울기
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
    


