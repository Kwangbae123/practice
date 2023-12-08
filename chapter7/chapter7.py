import numpy as np

x = np.random.rand(10, 1, 28, 28) # 갯수, 높이, 너비, 채널수
print(x.shape)
print(x.shape[0]) # 10개의 데이터중 첫번째 데이터에 접근

# im2col
# 3차원 입력데이터를 2차원 행렬로 바꾼다.
# 필터 적용 영역이 겹피면 원소수가 원래 블록의 원소수보다 많아진다. -> 메모리 소비로 이어진다.
