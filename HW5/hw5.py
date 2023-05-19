import numpy as np
import matplotlib.pyplot as plt

img1 = plt.imread('./data/warrior_a.jpg')
img2 = plt.imread('./data/warrior_b.jpg')

cor1 = np.load("./data/warrior_a.npy")
cor2 = np.load("./data/warrior_b.npy")


def compute_fundamental(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    F = None

    ### YOUR CODE BEGINS HERE

    from itertools import combinations

    numbers = list(range(n))
    subsets = list(combinations(numbers, 8))
    mse = None
    optimal_F = None
    for subset_indexes in subsets:
        A_array = []
        for i in subset_indexes:  # 4개의 포인터쌍에 대해서...
            A_array.append([x1[0][i] * x2[0][i], x1[1][i] * x2[0][i], x2[0][i],
                            x1[0][i] * x2[1][i], x1[1][i] * x2[1][i], x2[1][i],
                            x1[0][i], x1[1][i], 1])
            # A_array.append([x1[0][i] * x2[0][i], x1[0][i] * x2[1][i], x1[0][i], x1[1][i] * x2[0][i], x1[1][i] * x2[1][i], x1[1][i], x2[0][i], x2[1][i], 1])
        A_np_array = np.array(A_array)  # A 행렬 구하기
        ATA = np.dot(A_np_array.T, A_np_array)  # A^TA를 구하기
        ATA = ATA / ATA[-1][-1]  ## 마지막 항이 1이 되게끔 일반화

        eigenvalues, eigenvectors = np.linalg.eig(ATA)  # eigenvector/value 구하기
        smallest_eigenvalue_index = np.argmin(eigenvalues)  # 가장 작은 값의 index 구함
        smallest_eigenvector = eigenvectors[:, smallest_eigenvalue_index]  # 가장 작은 값의 eigenvector 획득
        F_temp = np.reshape(smallest_eigenvector, (3, 3))  # eigenvector를 통해 F획득

        U, S, V = np.linalg.svd(F_temp)  # svd 실행
        S[-1] = 0  # rank를 한 단계 낮춤
        # print(np.shape(U), np.shape(S), np.shape(V))
        S = np.diag(S)
        F_rank_2 = np.dot(np.dot(U, S), V)  # 획득한 F'
        F_rank_2 = F_rank_2/F_rank_2[-1][-1]


        # 모든 포인터 쌍에 대해서 오차 계산
        sum_of_squared_error = 0.0
        for i in range(n):
            error = np.dot(x2[:, i].T, np.dot(F_rank_2, x1[:, i].T))
            sum_of_squared_error += error ** 2
        print("mse : {0} \t sse : {1}".format(mse, sum_of_squared_error))
        # 최소오차일 경우 optimal_F를 업데이트
        if mse is None or mse > sum_of_squared_error:
            mse = sum_of_squared_error
            optimal_F = F_rank_2
    return optimal_F
    # build matrix for equations in Page 51
    # compute the solution in Page 51
    # constrain F: make rank 2 by zeroing out last singular value (Page 52)
    ### YOUR CODE ENDS HERE

    """ 
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i], x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i], x2[0, i], x2[1, i], 1]
    print(A)
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Enforce rank 2 constraint on F
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V
    print("=== : {}".format(F))
    return F
    """



def compute_norm_fundamental(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1, 0, -S1 * mean_1[0]], [0, S1, -S1 * mean_1[1]], [0, 0, 1]])
    x1 = T1 @ x1

    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2, 0, -S2 * mean_2[0]], [0, S2, -S2 * mean_2[1]], [0, 0, 1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1, x2)

    # reverse normalization
    F = T2.T @ F @ T1

    return F


def compute_epipoles(F):
    e1 = None
    e2 = None
    ### YOUR CODE BEGINS HERE
    U, S, V = np.linalg.svd(F)
    e1 = V[-1]
    e1 = e1/e1[2]

    # 에피폴 e2 계산
    U, S, V = np.linalg.svd(F.T)
    e2 = U[-1]
    e2 = e2/e2[2]
    ### YOUR CODE ENDS HERE

    return e1, e2


import cv2
import numpy as np


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)
    e1, e2 = compute_epipoles(F)

    # plt.scatter()
    # 이미지 크기 가져오기

    # 에피폴라인 그리기

    # 이미지1 위에 에피폴라인 그리기

    # 이미지2 위에 에피폴라인 그리기

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 이미지1 위에 에피폴라인 그리기

    ax1.imshow(img1)
    for i in range(cor1.shape[1]):
        x = cor1[0, i]
        y = cor1[1, i]
        line = np.dot(F.T, np.array([x, y, 1]))
        a, b, c = line
        ax1.plot([0, img1.shape[1]], [-c / b, -(c + a * img1.shape[1]) / b], 'r')

    # 이미지2 위에 에피폴라인 그리기
    ax2.imshow(img2)
    for i in range(cor2.shape[1]):
        x = cor2[0, i]
        y = cor2[1, i]
        line = np.dot(F, np.array([x, y, 1]))
        a, b, c = line
        ax2.plot([0, img2.shape[1]], [-c / b, -(c + a * img2.shape[1]) / b], 'g')

    plt.show()


draw_epipolar_lines(img1, img2, cor1, cor2)
