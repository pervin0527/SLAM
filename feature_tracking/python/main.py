import os
import cv2
import numpy as np

def main(image_path, num_frames):
    img_files = sorted(os.listdir(image_path))
    img_files = [os.path.join(image_path, file) for file in img_files][:num_frames]

    ## 종료 조건 설정
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    status, err = [], [] ## tracking 되고 있는지, 에러율은 얼마인지 저장.
    kpts, kpts_next = [], [] ## k번째 이미지의 특징점, k+1번째 이미지의 특징점
    for i in range(len(img_files) - 1):
        img = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)
        img_next = cv2.imread(img_files[i + 1], cv2.IMREAD_GRAYSCALE)

        img_vis = cv2.cvtColor(img_next, cv2.COLOR_GRAY2BGR) ## 시각화를 위한 컬러 이미지

        ## 특징점이 50개 미만인 경우 새롭게 특징점을 추출.
        if len(kpts) < 50:
            new_kpts = cv2.goodFeaturesToTrack(img, maxCorners=1000, qualityLevel=0.01, minDistance=15)
            
            ## 추적을 성공적으로 하고 있는 특징점을 유지하기 위해 신규 특징점을 extend로 추가.
            if new_kpts is not None:
                kpts.extend(new_kpts.reshape(-1, 2))

        ## Optical Flow
        if kpts:
            kpts = np.array(kpts, dtype=np.float32)
            kpts_next, status, err = cv2.calcOpticalFlowPyrLK(img, img_next, kpts, None, winSize=(21, 21), maxLevel=2, criteria=criteria)

            ## 시각화
            for j, (new, old) in enumerate(zip(kpts_next, kpts)):
                if status[j]:
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.circle(img_vis, (int(a), int(b)), 3, (0, 0, 255), -1)
                    direction = (c - a, d - b)
                    opposite_end = (int(a - direction[0]), int(b - direction[1]))
                    cv2.line(img_vis, (int(a), int(b)), opposite_end, (0, 255, 0), 1)

            cv2.imshow("Tracking", img_vis)
            cv2.waitKey(33)

            ## 상태가 유효하지 않은 특징점 제거
            kpts = [pt for pt, st in zip(kpts_next, status) if st]

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the folder containing images")
    parser.add_argument("num_frames", type=int, help="Number of frames to process")
    args = parser.parse_args()

    main(args.image_path, args.num_frames)
