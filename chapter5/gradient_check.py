# 기울기를 구하는 방법은 1. 수치미분, 2. 해석적으로 수식을 풀어 구하는 방법
# 수치미분은 느리지만 구현하기가 쉽고 버그가 없다.
# 오차역전파는 빠르지만 구현하기다 어렵고 버그가 있을수 있다.
# 따라서 gradient_check을 통해 두방식을 구한 기울기가 일치하는지 확인을 한다.
import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

#각 가중치의 차이의 절대값을 구한 후 , 그 절대값들의 평균을 낸다.
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key +" : {0}".format(diff))