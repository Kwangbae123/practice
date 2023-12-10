# validation dataset으로 분할 하기 P.222
import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist
from utils.util import shuffle_dataset

(x_train, t_train), (x_test, t_test) = load_mnist()

# 훈련데이터를 섞는다.
x_train, t_train = shuffle_dataset(x_train, t_train)

# 20를 validation data로 분할 한다.
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]