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
x_sobel = np.array([ 1, 0, -1, 2, 0, -2, 1, 0, -1])
y_sobel = np.array([ 1, 2, 1, 0, 0, 0, -1, -2, -1])
x_sobel = x_sobel.reshape((3, 3))/8.0
y_sobel = y_sobel.reshape((3, 3))/8.0

# 아래 상수는 divide by zero를 방지하기 위한 적당히 작은 수이다.
QUITELY_SMALL = 0.00000000001
def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    # x와 y의 방향에 대해서 각각 sobel 필터를 적용한다.
    x_gradient_img = convolve2d(img, x_sobel)
    y_gradient_img = convolve2d(img, y_sobel)

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
    # 각도에 따라서 각 픽셀에서 남길 만한 값을 제외하고는 제거한다.
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
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
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

def double_thresholding(img):
    diff = max(img) - min(img)
    T_high = min(img) + diff * 0.15
    T_low = min(img) + diff * 0.03

    res = img[:]

    img[img > T_high] = 255
    img[T_high > img > T_low] = 80
    img[T_low > img] = 0
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
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
    diff =
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    pass
    return res



