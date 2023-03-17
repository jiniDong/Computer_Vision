from PIL import Image
import numpy as np
import math


def boxfilter(n):
    if n <= 0 or n % 2 == 0:  # n이 조건에 맞지 않을 경우 Assertion Erorr를 발생시킵니다.
        raise AssertionError('Dimension must be odd')

    box_entry = 1.0 / n ** 2
    filter = np.full((n, n), box_entry)
    return filter


# print(boxfilter(3))
# print(boxfilter(4))
# print(boxfilter(7))

def gauss1d(sigma):
    width: int = int(math.ceil(6 * sigma))  # 6*sigma의 다음 정수로 width를 설정
    if width % 2 == 0:  # 다음 정수가 짝수라면 +1 하여 홀수로 변경
        width += 1

    center = math.floor(width / 2)  # 가운데 index를 획득
    gaussian = np.zeros((1, width)).flatten()  # 반환할 1D Numpy Array 생성
    for i in range(width):
        gaussian[i] = i - center  # Hint:를 토대로 x값을 입력
    for i in range(width):
        gaussian[i] = math.exp(-gaussian[i] ** 2 / (2 * sigma ** 2))  # x값에 따라 gaussian값 획득 및 입력

    # 아래는 normalize, 전체 합으로 각 엔트리를 나눈다.
    kernal_sum = np.sum(gaussian)
    for i in range(width):
        gaussian[i] /= kernal_sum
    return gaussian


# 결과 출력
print("sigma = 0.3\n{}".format(gauss1d(0.3)))
print("sigma = 0.5\n{}".format(gauss1d(0.5)))
print("sigma = 1.0\n{}".format(gauss1d(1.0)))
print("sigma = 2.0\n{}".format(gauss1d(2.0)))


def gauss2d(sigma):
    return np.outer(gauss1d(sigma), gauss1d(sigma))

print("gauss2d(0.5) = \n{}".format(gauss2d(0.5)))
print("gauss2d(1.0) = \n{}".format(gauss2d(1.0)))


