import numpy as np

#step_function 구현
def step_function(x):
    return np.array(x > 0, dtype=np.int)

#sigmoid 함수 구현
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#ReLU 함수 구현
def relu(x):
    return np.maximum(0, x) #maximun 두 입력중 큰값을 선택해 반환

# Softmax 함수 구현
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

#오차제곱합(sum of squares for error, SSE)
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# Cross Entropy Error
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

