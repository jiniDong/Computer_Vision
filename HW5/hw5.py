import numpy as np
import matplotlib.pyplot as plt

img1 = plt.imread('./data/graffiti_a.jpg')
img2 = plt.imread('./data/graffiti_b.jpg')

cor1 = np.load("./data/graffiti_a.npy")
cor2 = np.load("./data/graffiti_b.npy")


def compute_fundamental(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)

    ### YOUR CODE BEGINS HERE

    from itertools import combinations

    numbers = list(range(n))
    subsets = list(combinations(numbers, 8))
    mse = None
    optimal_F = None


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
    return F



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
    e1 = e1 / e1[2]

    # 에피폴 e2 계산
    U, S, V = np.linalg.svd(F.T)
    e2 = U[-1]
    e2 = e2 / e2[2]
    ### YOUR CODE ENDS HERE

    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)
    e1, e2 = compute_epipoles(F)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    colors = []
    for i in range(cor1.shape[1]):
        colors.append(np.random.rand(3))

    # img1에 대해서
    ax1.imshow(img1)
    index = 0
    for i in range(cor1.shape[1]):
        # 점 찍기
        ax1.plot(cor1[0, i], cor1[1, i], 'o', color=colors[index])
        # line 그리기
        x = np.array([cor1[0, i], e1[0]])
        y = np.array([cor1[1, i], e1[1]])
        ax1.plot(x, y, color=colors[index])
        index += 1

    # img2에 대해서
    ax2.imshow(img2)
    index = 0
    for i in range(cor2.shape[1]):
        # 점 찍기
        ax2.plot(cor2[0, i], cor2[1, i], 'o', color=colors[index])
        # line 그리기
        x = np.array([cor2[0, i], e2[0]])
        y = np.array([cor2[1, i], e2[1]])
        ax2.plot(x, y, color=colors[index])
        index += 1

    plt.show()


draw_epipolar_lines(img1, img2, cor1, cor2)
