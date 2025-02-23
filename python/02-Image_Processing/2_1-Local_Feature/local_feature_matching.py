import os
import cv2
import numpy as np

def vis_imgs(left_img_path, right_img_path):
    """ OpenCV를 이용해 두 개의 흑백 이미지를 시각화 """
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"Left Image Shape: {left_img.shape}, Right Image Shape: {right_img.shape}")

    # 두 개의 이미지를 가로로 연결하여 표시
    combined_img = np.hstack((left_img, right_img))
    
    cv2.imshow("Left and Right Images", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vis_img_feature_extract(img_path):
    """ ORB 특징점 검출 후 OpenCV로 시각화 """
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create(1000)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    print(f"Detected {len(keypoints)} keypoints")
    if keypoints:
        sample_keypoint = keypoints[0]
        print(f"첫 번째 KeyPoint:")
        print(f" - 좌표 (x, y): ({sample_keypoint.pt[0]:.2f}, {sample_keypoint.pt[1]:.2f})")
        print(f" - 크기 (size): {sample_keypoint.size:.2f}")
        print(f" - 방향 (angle): {sample_keypoint.angle:.2f}")
        print(f" - 응답값 (response): {sample_keypoint.response:.6f}")
        print(f" - 옥타브 (octave): {sample_keypoint.octave}")
        print(f" - 클래스 ID (class_id): {sample_keypoint.class_id}")

    # 특징점 그리기
    img_kp = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("ORB Features", img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if descriptors is not None:
        print(f"Descriptors shape: {descriptors.shape}")
        print(f"Descriptors dtype: {descriptors.dtype}")


def vis_feature_matching(left_img_path, right_img_path):
    """ ORB 특징점 매칭 후 OpenCV로 시각화 """
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create(1000)
    left_kps, left_descs = orb.detectAndCompute(left_img, None)
    right_kps, right_descs = orb.detectAndCompute(right_img, None)

    # Brute-Force 매처 생성
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 특징점 매칭
    matches = bf.match(left_descs, right_descs)
    print(type(matches), len(matches))

    # 첫 번째 매칭 정보 출력
    if matches:
        match = matches[0]
        print(f"첫 번째 매칭 정보:")
        print(f"  - queryIdx (왼쪽 이미지 특징점 인덱스): {match.queryIdx}")
        print(f"  - trainIdx (오른쪽 이미지 특징점 인덱스): {match.trainIdx}")
        print(f"  - 매칭 거리(distance): {match.distance}")

    # 거리순 정렬 (가까운 거리순으로 정렬)
    matches = sorted(matches, key=lambda x: x.distance)

    # 매칭 결과 시각화
    matched_img = cv2.drawMatches(left_img, left_kps, right_img, right_kps, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow("Feature Matching", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    DATA_DIR = "/home/pervinco/Datasets/KITTI/dataset/sequences/00"
    LEFT_IMG_DIR = f"{DATA_DIR}/image_0"
    RIGHT_IMG_DIR = f"{DATA_DIR}/image_1"
    
    left_img_files = os.listdir(LEFT_IMG_DIR)
    right_img_files = os.listdir(RIGHT_IMG_DIR)
    
    print(f"Number of images: {len(left_img_files)}, {len(right_img_files)}")

    left_img_path = os.path.join(LEFT_IMG_DIR, left_img_files[0])
    right_img_path = os.path.join(RIGHT_IMG_DIR, right_img_files[0])

    vis_imgs(left_img_path, right_img_path)
    vis_img_feature_extract(left_img_path)
    vis_feature_matching(left_img_path, right_img_path)


if __name__ == "__main__":
    main()
