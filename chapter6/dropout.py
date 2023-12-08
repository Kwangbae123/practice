# Drop out class 구현 P.219
import numpy as np
class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg = True):
        if train_flg:
            # 훈련시 self.mask에 삭제할 뉴런을 false로 표현 / slef.mask는 x와 형상이 같은 배열
            # dropout_ratio보다 큰 원소만 True로 설정
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        # 역전파시 순전파때 통과시키는 뉴런은 통과 / 통과하지 않은 뉴런은 억제
        return dout * self.mask