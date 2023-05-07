import numpy as np
import cv2
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    largest_set = []
    for i in range(10):  # repeat ten times
        rand = random.randrange(0, len(matched_pairs))  # generate random number
        choice = matched_pairs[rand]
        orientation = (keypoints1[choice[0]][3] - keypoints2[choice[1]][3]) % (
                2 * math.pi)  # calculation first-orientation
        scale = keypoints2[choice[1]][2] / keypoints1[choice[0]][2]  # calculation first-scale ratio
        temp = []
        for j in range(len(matched_pairs)):  # calculate the number of all cases
            if j is not rand:
                # calculation second-orientation
                orientation_temp = (keypoints1[matched_pairs[j][0]][3] -
                                    keypoints2[matched_pairs[j][1]][3]) % (2 * math.pi)
                # calculation second-scale-ratio
                scale_temp = keypoints2[matched_pairs[j][1]][2] /\
                             keypoints1[matched_pairs[j][0]][2]
                # check degree error +=30degree
                if (orientation - orient_agreement / 6) < orientation_temp \
                        < (orientation + orient_agreement / 6):
                    # check scale error +- 50%
                    if scale - scale * scale_agreement < scale_temp\
                            < scale + scale * scale_agreement:
                        temp.append([i, j])
        if len(temp) > len(largest_set):
            largest_set = temp
    for i in range(len(largest_set)):
        largest_set[i] = (matched_pairs[largest_set[i][1]][0],
                          matched_pairs[largest_set[i][1]][1])

    ## END
    assert isinstance(largest_set, list)
    return largest_set


def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    ## the following is just a placeholder to show you the output format
    y1 = descriptors1.shape[0]  # 해당 코드는 descriptor1 의 요소들을 순회하기 위해서 필요함
    y2 = descriptors2.shape[1]  # 해당 코드는 descriptor2 의 요소들을 순회하기 위해서 필요함
    temp = np.zeros(y2)
    matched_pairs = []

    for i1 in range(y1):
        for i2 in range(y2):
            temp[i2] = math.acos(np.dot(descriptors1[i1], descriptors2[i2]))
        compare = sorted(range(len(temp)), key=lambda k: temp[k])
        if (temp[compare[0]] / temp[compare[1]]) < threshold:
            matched_pairs.append([i1, compare[0]])
    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    projection이 되는 point들을 return하는...
    homogeneous coordinate 로 차원을 변경한 후, 2D로 변경하는 작업
    divide by zero 방지를 위해 마지막 차원에 1e10을 대입시키는 것을 고려
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    # 결과를 위한 배열을 생성
    xy_points_out = np.ones((np.shape(xy_points)[0], 3))
    # print(np.shape(xy_points_out))
    # print(np.shape(xy_points))
    # 배열의 모든 pointer에 대해서
    for i in range(len(xy_points)):
        # 각 pointer를 h와 dot연산하여 가져온다.
        xy_points_out[i] = np.dot(h, np.append(xy_points[i], 1))
        # 마지막 차원의 값에 따라서 x, y값 변경하기 마지막 값이 0일 경우에는 대체값을 적용
        if xy_points_out[i][2] != 0:
            xy_points_out[i] = xy_points_out[i] / xy_points_out[i][2]
        else:
            xy_points_out[i] = xy_points_out[i] / 1e10
    # 마지막 축은 제외한 체 x,y point 반환
    xy_points_out = xy_points_out[:, :2]
    # END
    return xy_points_out


def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations. 몇 번 랜덤하게 뽑는가?
        tol: float inlier의 기준
    Outputs:
        h: The final homography matrix.

    src에서 ref로 투영시켜 거리를 측정, tolerence보다 작을 경우 inlier로 취급
    가장 inlier가 많은 것을 seclect

    어색한 boundary는 괜찮아요
    """

    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol * 1.0

    # START
    # 네 개 뽑아, homogeneous만들고, 다 투영시켜서 inlier숫자 선별, inlier가 가장 많은 것을 선택하여 반환
    h = np.identity(3) # h 초기화
    num_inliers  = 0 # 카운트할 iniler들
    h_temp = np.array([])
    for stage in range(num_iter):
        # 랜덤히게 4 개의 index를 선택
        selected_index = random.sample(range(len(xy_src)), k=4)
        # A 배열 만들기
        A_array = []
        for i in selected_index:
            A_array.append([xy_src[i][0], xy_src[i][1], 1, 0, 0, 0,
                            -xy_src[i][0] * xy_ref[i][0], -xy_ref[i][0] * xy_src[i][1], xy_ref[i][0]])
            A_array.append([0, 0, 0, xy_src[i][0], xy_src[i][1], 1,
                            -xy_ref[i][1] * xy_src[i][0], -xy_ref[i][1] * xy_src[i][1], xy_ref[i][1]])
        A = np.array(A_array)
        # 가장 작은 eigenvalue와 그에 해당하는 eigenvector를 구한다.
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.transpose(A), A))

        min_value = float('inf')
        min_index = -1
        for i in range(len(eigenvalues)):
            if eigenvalues[i] < min_value:
                min_value = eigenvalues[i]
                min_index = i
        # 찾아낸 eigenvelue로 h 생성
        h_temp = eigenvectors[min_index]
        h_temp = h_temp.reshape(3, 3)
        h_temp = h_temp/h_temp[2][2]
        # 생성된 h를 이용해서 xy_src의 좌표들을 투영
        xy_proj = KeypointProjection(xy_src, h)

        # inlier들을 골라냄
        temp_num_inlier = 0
        for i in range(len(xy_proj)):
            if np.linalg.norm(xy_proj[i] - xy_ref[i]) < tol:
                temp_num_inlier += 1
        # 가장 적은 inlier일 경우 h를 변경 아니라면 폐기
        if temp_num_inlier > num_inliers :
            num_inliers = temp_num_inlier
            h = h_temp
    """
    # 찾아낸 4 개의 좌료를 통해 homogeneous matrix를 구한다.
    # projection한 것과 차이를 구해서 tolerence 이내에서 inlier를 구한다.
    # outlier가 가장 작은 것을 선정
    """

    # END
    # homography의 마지막 요소는 1이어야 하므로
    h = h/h[2][2]
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)

    h, _ = cv2.findHomography(xy_src, xy_ref)
    print(h)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website

    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
