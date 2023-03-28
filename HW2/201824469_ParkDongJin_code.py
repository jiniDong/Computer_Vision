from PIL import Image
import numpy as np
import math


def boxfilter(n):
    if n <= 0 or n % 2 == 0:  # n이 조건에 맞지 않을 경우 Assertion Error를 발생시킵니다.
        raise AssertionError('Dimension must be odd')

    # 각 픽셀에 들어갈 값을 계산한다.
    # 모두 합해서 1이어야 하고, 모든 픽셀의 가중치가 같아야 한다.
    box_entry = 1.0 / n ** 2
    filter = np.full((n, n), box_entry)
    return filter


print(boxfilter(3))
# print(boxfilter(4))
print(boxfilter(7))


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
# outer product로 2d 가우시안 필터를 구한다.

print("gauss2d(0.5) = \n{}".format(gauss2d(0.5)))
print("gauss2d(1.0) = \n{}".format(gauss2d(1.0)))


def convolove2d(array, filter):
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
    for i in range(padding_width, np.shape(zero_padding)[0] - padding_width):
        for j in range(padding_width, np.shape(zero_padding)[1] - padding_width):
            subpart = zero_padding[i - padding_width:i + padding_width + 1,
                      j - padding_width:j + padding_width + 1]
            zero_padding[i][j] = np.sum(subpart * flipped_filter)
    # 컨벌루젼한 값을 리턴
    return zero_padding[padding_width:-padding_width, padding_width:-padding_width]


def gaussconvolve2d(array, sigma):
    return convolove2d(array, gauss2d(sigma))


# 파일 불러오기 및 가우시안 효과 적용
dog_img = Image.open('hw2_image/2b_dog.bmp')
dog_img_grey = dog_img.convert('L')
dog_img_array = np.asarray(dog_img_grey)
blurred_dog_array = np.asarray(gaussconvolve2d(dog_img_array, 3.0))

# 두 개의 원본과 하나의 결과물 이미지 확인
dog_img.show()
dog_img_grey.show()
Image.fromarray(blurred_dog_array).show()

# Part 2: Hybrid Image
# 1.

# 사용할 이미지를 불러오기
dog_img_img = Image.open('hw2_image/2b_dog.bmp')
cat_img = Image.open('hw2_image/2a_cat.bmp')

# 이미지를 배열로 변환
dog_img_img_array = np.asarray(dog_img_img)
cat_img_array = np.asarray(cat_img)

# 변환된 배열의 shape를 가져온다.
dog_img_height, dog_img_width, rgb = np.shape(dog_img_img_array)

# 각 픽셀의 RGB값을 가져온다.
dog_img_img_array_r = dog_img_img_array[:, :, 0].reshape(dog_img_height, dog_img_width)
dog_img_img_array_g = dog_img_img_array[:, :, 1].reshape(dog_img_height, dog_img_width)
dog_img_img_array_b = dog_img_img_array[:, :, 2].reshape(dog_img_height, dog_img_width)

# 각 RGB배열에 가우시안 컨벌루젼을 진행한다.
blur_sigma = 5.0
dog_img_img_array_r_blurred = gaussconvolve2d(dog_img_img_array_r, blur_sigma)
dog_img_img_array_g_blurred = gaussconvolve2d(dog_img_img_array_g, blur_sigma)
dog_img_img_array_b_blurred = gaussconvolve2d(dog_img_img_array_b, blur_sigma)

# 결과를 저장할 배열을 만든다.
dog_img_img_blurred = np.zeros((dog_img_height, dog_img_width, rgb))

# 각 픽셀의 RGB값을 병합한다.
for row in range(dog_img_height):
    for pixel in range(dog_img_width):
        dog_img_img_blurred[row][pixel] = np.array([dog_img_img_array_r_blurred[row][pixel],
                                                    dog_img_img_array_g_blurred[row][pixel],
                                                    dog_img_img_array_b_blurred[row][pixel]])

# 병합된 이미지를 보여준다.
Image.fromarray(dog_img_img_blurred.astype(np.uint8)).show()

# 2.

# 변환된 배열의 shape를 가져온다.
cat_height, cat_width, rgb = np.shape(cat_img_array)

# 각 픽셀의 RGB값을 가져온다.
cat_img_array_r = cat_img_array[:, :, 0].reshape(cat_height, cat_width)
cat_img_array_g = cat_img_array[:, :, 1].reshape(cat_height, cat_width)
cat_img_array_b = cat_img_array[:, :, 2].reshape(cat_height, cat_width)

# 각 RGB에 대해서 가우시안 효과 적용
sharp_sigma = 5.0
cat_img_array_r_blurred = gaussconvolve2d(cat_img_array_r, sharp_sigma)
cat_img_array_g_blurred = gaussconvolve2d(cat_img_array_g, sharp_sigma)
cat_img_array_b_blurred = gaussconvolve2d(cat_img_array_b, sharp_sigma)

# 결과물을 저장할 배열을 만든다.
cat_img_sharpened = np.zeros((dog_img_height, dog_img_width, rgb))

# 각 픽셀에 원본에서 가우시안효과를 뺀 값을 저장한다.
for row in range(cat_height):
    for pixel in range(cat_width):
        cat_img_sharpened[row][pixel] = np.array([cat_img_array_r[row][pixel]
                                                         - cat_img_array_r_blurred[row][pixel],
                                                         cat_img_array_g[row][pixel]
                                                         - cat_img_array_g_blurred[row][pixel],
                                                         cat_img_array_b[row][pixel]
                                                         - cat_img_array_b_blurred[row][pixel]])

# 고주파 이미지를 보여주기 위한 보정값 128을 각 픽셀에 더한 후 이미지 보여주기
cat_img_show = cat_img_sharpened + 128
Image.fromarray(cat_img_show.astype(np.uint8)).show()

# 3

# 두 개의 이미지를 더한다. 픽셀 값의 오버플로를 막기 위해 2로 나눈다.
hybrid = (cat_img_sharpened + dog_img_img_blurred)/2
np.where(hybrid > 255, 255, hybrid)
np.where(hybrid < 0, 0, hybrid)
Image.fromarray(hybrid.astype(np.uint8)).show()