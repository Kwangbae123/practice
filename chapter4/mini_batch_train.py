import numpy as np
import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist
from two_layer_net_base import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#하이퍼 파라미터
iters_num = 1000 # 반복횟수
train_size = x_train.shape[0]
batch_size = 100 # 미니배치 크기
learning_rate = 0.1 # 학습률

train_loss_list =[]
train_acc_list = []
test_acc_list = []

# 1 epoch당 반복수
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size= 784, hidden_size= 50, output_size= 10)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)

    #매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # print(train_loss_list)

    # 1 epoch당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train,t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc : {0} test acc : {1}".format(train_acc, test_acc))


