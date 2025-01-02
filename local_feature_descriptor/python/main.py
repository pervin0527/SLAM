import os
import cv2

def draw_keypoints_and_matches(left_img, right_img, left_kp, right_kp, matches):
    # Draw keypoints on the images
    left_kp_image = cv2.drawKeypoints(left_img, left_kp, None, color=(0, 255, 0))
    right_kp_image = cv2.drawKeypoints(right_img, right_kp, None, color=(0, 255, 0))

    # Draw matches
    match_image = cv2.drawMatches(left_img, left_kp, right_img, right_kp, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return left_kp_image, right_kp_image, match_image


def main():
    save_path = "./result"
    os.makedirs(save_path, exist_ok=True)

    data_path = "/home/pervinco/Datasets/KITTI/dataset/sequences/00/"
    left_img_path = os.path.join(data_path, "image_0")
    right_img_path = os.path.join(data_path, "image_1")
    print(left_img_path, right_img_path)

    left_img_files = sorted(os.listdir(left_img_path))
    right_img_files = sorted(os.listdir(right_img_path))
    print(len(left_img_files), len(right_img_files))
    assert len(left_img_files) == len(right_img_files)

    ## keypoint, descriptor 추출기 생성
    detector = cv2.SIFT_create()

    ## brute force matcher 생성
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # FLANN matcher 생성
    search_params = dict(checks=50)  # 검색 매개변수, 검색할 트리의 수
    index_params = dict(algorithm=1,  # FLANN_INDEX_KDTREE
                        trees=5)      # KDTree의 트리 수
    knn_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    for left_file, right_file in zip(left_img_files, right_img_files):
        left_file = os.path.join(left_img_path, left_file)
        right_file = os.path.join(right_img_path, right_file)

        left_img = cv2.imread(left_file, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_file, cv2.IMREAD_GRAYSCALE)

        ## keypoint : tuple, (1000, )
        ## descriptor : numpy.ndarray, (1000, 32)
        # left_kp, left_desc = detector.detectAndCompute(left_img, None)
        # right_kp, right_desc = detector.detectAndCompute(right_img, None)

        left_kp, left_desc = detector.detectAndCompute(left_img, None)
        right_kp, right_desc = detector.detectAndCompute(right_img, None)

        # 디스크립터를 float32 형식으로 변환
        left_desc = left_desc.astype('float32')
        right_desc = right_desc.astype('float32')

        # matches = bf_matcher.match(left_desc, right_desc)
        # matches = sorted(matches, key=lambda x: x.distance)

        matches = knn_matcher.knnMatch(left_desc, right_desc, k=2)
        # SNN 방식 적용
        ratio_thresh = 0.8
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        left_kp_image, right_kp_image, match_image = draw_keypoints_and_matches(left_img, right_img, left_kp, right_kp, good_matches)

        # cv2.imshow("left_kp_image", left_kp_image)
        # cv2.imshow("right_kp_image", right_kp_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        left_save_path = os.path.join(save_path, f"left_kp_{os.path.basename(left_file)}")
        right_save_path = os.path.join(save_path, f"right_kp_{os.path.basename(right_file)}")
        match_save_path = os.path.join(save_path, f"match_{os.path.basename(left_file)}")

        cv2.imwrite(left_save_path, left_kp_image)
        cv2.imwrite(right_save_path, right_kp_image)
        cv2.imwrite(match_save_path, match_image)

        break

if __name__ == "__main__":
    main()