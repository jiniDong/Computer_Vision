from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""
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

def gauss2d(sigma):
    return np.outer(gauss1d(sigma), gauss1d(sigma))
# outer product로 2d 가우시안 필터를 구한다.

def convolve2d(array, filter):
    padding_width = int(len(filter) / 2)  # filter에 따른 padding 크기
    zero_padding = np.zeros((np.shape(array)[0] + padding_width * 2,
                             np.shape(array)[1] + padding_width * 2))
    # padding한 크기에 맞게 새로운 np.array를 생성

    # convolution연산을 위해서 filter를 축의 방향에 대해서 뒤집는다.
    flipped_filter = np.flip(filter, axis=0)
    flipped_filter = np.flip(flipped_filter, axis=1)
    # 모든 픽셀에 대해서 paading을 제외한 부분에 array를 집어넣어 준다.
    for i in range(padding_width, np.shape(zero_padding)[0] - padding_width):
        for j in range(padding_width, np.shape(zero_padding)[1] - padding_width):
            zero_padding[i][j] = array[i - padding_width][j - padding_width]

    # padding된 이미지의 각 픽셀을 순회하면서 각 픽셀에 컨벌루젼한 값을 입력한다.

    convolved = np.zeros(np.shape(array))

    for i in range(padding_width, np.shape(zero_padding)[0] - padding_width):
        for j in range(padding_width, np.shape(zero_padding)[1] - padding_width):
            subpart = zero_padding[i - padding_width:i + padding_width + 1,
                      j - padding_width:j + padding_width + 1]
            convolved[i-padding_width][j-padding_width] = np.sum(subpart * flipped_filter)
    # 컨벌루젼한 값을 리턴
    return convolved

def gaussconvolve2d(array, sigma):
    return convolve2d(array, gauss2d(sigma))

#sobel연산을 위한 필터를 제작한다. normalization을 위해 8로 나눈다.
X_SOBEL = np.array([1, 0, -1, 2, 0, -2, 1, 0, -1])
Y_SOBEL = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1])
X_SOBEL = X_SOBEL.reshape((3, 3)) / 8.0
Y_SOBEL = Y_SOBEL.reshape((3, 3)) / 8.0

# 아래 상수는 divide by zero를 방지하기 위한 적당히 작은 수이다.
QUITELY_SMALL = 0.00000000001
def sobel_filters(img):
    # x와 y의 방향에 대해서 각각 sobel 필터를 적용한다.
    x_gradient_img = convolve2d(img, X_SOBEL)
    y_gradient_img = convolve2d(img, Y_SOBEL)

    # G를 계산한다.
    G = np.hypot(x_gradient_img, y_gradient_img)
    # G의 각 원소를 0과 255사이로 매핑한다.
    G = (G / np.max(G)) * 255

    # np.vectorize 함수는 np.array의 모든 원소에 특정 함수를 적용하는 것을 쉽게 만든다.
    vectorized_arctan = np.vectorize(np.arctan)
    # arctan의 분모에 들어갈 x값이 0일 경우 divide by zero가 발생할 수 있으므로 0을 적당히 작은 값으로 치환한다.
    x_gradient_img[x_gradient_img == 0] = QUITELY_SMALL
    # np.arctan를 이용해서 theta 배열을 구한다.
    theta = vectorized_arctan(y_gradient_img / x_gradient_img)
    return (G, theta)

def remain_max(array, x, y, angle):
    """
    그림 배열 array의 x(column), y(row)위치에서 angle에 따라 해당 픽셀에 할당되어야 하는 값을 반환한다.
    할당되어야 하는 값이란 굵게 표시된 edge의 최댓값만을 남기기 위해 픽셀에 적용해야 하는 값이다.

    :param array:적용할 이미지 배열
    :param x:x값, 즉 column number
    :param y:y값, 즉 row number
    :param angle: 해당 픽셀에서의 gradient 방향
    :return: 해당 픽셀에 최고 edge만을 남기기 위해서 적용되어야 하는 값,
    해당 방향으로의 주변픽셀을 확인하여 자신이 최댓값일 경우 남기고 아닐 경우 0으로 치환
    """
    ret = 0
    # -2|2 => 90도, 1 => 45도, 0 => 0도, -1 => 135도
    if angle == -2 or angle == 2:
        if array[y][x] < max(array[y+1][x], array[y][x], array[y-1][x]):
            ret = 0
        else:
            ret =  array[y][x]
    if angle == -1:
        if array[y][x] < max(array[y+1][x-1], array[y][x], array[y-1][x+1]):
            ret =  0
        else:
            ret =  array[y][x]
    if angle == 0:
        if array[y][x] < max(array[y][x+1], array[y][x], array[y][x-1]):
            ret = 0
        else:
            ret = array[y][x]
    if angle == 1:
        if array[y][x] < max(array[y+1][x+1], array[y][x], array[y-1][x-1]):
            ret = 0
        else:
            ret = array[y][x]
    return ret

def non_max_suppression(G, theta):
    # 라디안으로 들어온 theta를 각도로 변환시킨다.
    theta_degree = theta * 180 / np.pi
    # 변환된 각도를 0, 45, 90, 135중에서 가장 가까운 각도에 mapping시킨다.
    # 22.5를 더하는 것은
    theta_mapped = (theta_degree + 22.5) // 45
    """
    theta_mapped의 각 값을 해석하면 아래와 같다.
    -2|2 => 90도, 1 => 45도, 0 => 0도, -1 => 135도
    해석에 따라 주변 픽셀을 비교하여 non_max_suppression을 수행한다.
    """
    # 각도에 따라서 각 픽셀에서 남길 만한 값을 제외하고는 제거한다.
    res = np.zeros(np.shape(G))
    for row in range(1, len(G)-1):
        for col in range(1, len(G[1])-1):
            res[row][col] = remain_max(G, col, row, theta_mapped[row][col])
    # 결과를 반환한다.
    return res

#
STRONG_EDGE_INTENSE = 255
WEAK_EDGE_INTENSE = 80
def apply_double_threshold(entry, T_high, T_low):
    """
    :param entry: Threshold 적용할 값
    :param T_high: Strong Edge의 Threshold
    :param T_low: Weak Edge의 Threshold
    :return: Entry의 값을 보고, T_high보다 높으면 STRONG_EDGE_INTENSE,
     T_low와 T_high사이이면 WEAK_EDGE_INTENSE,
     T_low보다 작으면 0.
    """
    if entry >= T_high:
        return STRONG_EDGE_INTENSE
    elif T_high > entry >= T_low:
        return WEAK_EDGE_INTENSE
    else:
        return 0
def double_thresholding(img):
    # diff와 문제에서 제시된 threshold를 구한다.
    diff = img.max()
    T_high = img.min() + diff * 0.15
    T_low = img.min() + diff * 0.03

    # 결과를 기록하고 반환할 res 배열을 img에서 복사한다.
    res = img[:]

    # 각 픽셀에 threshold를 적용한 값을 입력하기 위해 vectorize 생성 후 threshold 적용
    vectorized_threshold = np.vectorize(apply_double_threshold)
    res = vectorized_threshold(img, T_high, T_low)

    # 완성된 결과를 반환
    return res

def dfs(img, res, i, j, visited=[]):
    # 호출된 시점의 시작점 (i, j)은 최초 호출이 아닌 이상 
    # strong 과 연결된 weak 포인트이므로 res에 strong 값을 준다
    res[i, j] = 255

    # 이미 방문했음을 표시한다
    visited.append((i, j))

    # (i, j)에 연결된 8가지 방향을 모두 검사하여 weak 포인트가 있다면 재귀적으로 호출
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    # 반환할 배열을 생성
    res = np.zeros(np.shape(img))
    # 방문지 기록
    visited = []
    # 순회 인덱스를 찾기 위한 width, height 구하기
    width = len(img)
    height = len(img[0])

    # 모든 픽셀(모서리는 out of range를 방어하기 위해 제외)을 순회하며
    # strong edge에 대해서 dfs 수행
    for i in range(1, width-1):
        for j in range(1, height-1):
            if img[i][j] == 255:
                dfs(img, res, i, j, visited)
    return res


#sobel연산을 위한 필터를 제작한다. normalization을 위해 8로 나눈다.
x_sobel = np.array([ 1, 0, -1, 2, 0, -2, 1, 0, -1])
y_sobel = np.array([ 1, 2, 1, 0, 0, 0, -1, -2, -1])
x_sobel = x_sobel.reshape((3, 3))/8.0
y_sobel = y_sobel.reshape((3, 3))/8.0

def main():
    # 1. noise reduction
    # 이구아나 사진을 불러온다.
    iguana_img = Image.open('iguana.bmp')
    # 사진을 흑백처리한다.
    iguana_img_grey = iguana_img.convert('L')
    # 이미지를 np 배열형태로 변형한다.
    iguana_array_grey = np.asarray(iguana_img_grey)
    # 가우시안 필터 효과를 적용한다.
    iguana_array_grey_blur = np.uint8(gaussconvolve2d(iguana_array_grey, 1.6))
    # 배열을 이미지로 변경하여 보여준다.
    Image.fromarray(iguana_array_grey_blur.astype(np.uint8)).show()

    # 2. Finding the intensity gradient of image
    mapped_sum_gradient_iguana, theta_iguana = sobel_filters(iguana_array_grey_blur)
    Image.fromarray(mapped_sum_gradient_iguana.astype(np.uint8)).show()

    # 3. Non-Maximum Suppression
    non_max_suppressed_iguana = non_max_suppression(mapped_sum_gradient_iguana, theta_iguana)
    Image.fromarray(non_max_suppressed_iguana.astype(np.uint8)).show()

    # 4. Double threshold
    double_threshold_iguana = double_thresholding(non_max_suppressed_iguana)
    Image.fromarray(double_threshold_iguana.astype(np.uint8)).show()

    # 5. Edge Tracking by hysteresis
    hysteresised = hysteresis(double_threshold_iguana)
    Image.fromarray(hysteresised.astype(np.uint8)).show()

main()