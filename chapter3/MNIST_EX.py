#MNIST 데이터셋 0~9까지 숫자 이미지
import pickle
import chapter3
import numpy as np
import sys, os
from PIL import Image

sys.path.append((os.pardir))
from mnist import load_mnist

def img_show(img):
    pil_img = Image.fromarray((np.uint8(img))) #이미지를 PIL용 데이터 객체로 변환
    pil_img.show()

# MNIST 데이터 가져오기 / [load_mnist는 normalize = 0~1.0 사이의 값으로 정규화할지 / flatten = 1차원 배열로 만들지 / one_hot_label = 원핫인코딩 형태로 저장할지]
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label) # 숫자 '5'사진 출력

print(img.shape) #1차원 배열로 이루어진 이미지
img = img.reshape(28, 28) #원래 이미지 모양으로 변경
print(img.shape)

img_show(img)

# MNIST 데이터 신경망 구현
# 입력층 28*28= 784 / 1층 = 50(임의) / 2층 = 100(임의) / 출력층 = 10(0~9)

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True) # 전처리 작업으로 종규화를 수행한다.
    return x_test, t_test

def init_network():
    with open("../sample_weight.pkl", 'rb') as f: # 학습된 가중치의 매개변수를 읽는다. 파일에 가중치, bias의 변수가 저장되어있음
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = chapter3.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = chapter3.sigmoid(a2)
    a3 = np.dot(z2, W3)+ b3
    y = a3
    return y

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size] # batch_size 크기 만큼 데이터를 묶는다. 
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) # 확률이 가장높은 원소의 인덱스를 얻는다. = 예측결과
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy : {0}".format(float(accuracy_cnt) / len(x)))