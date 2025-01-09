// ./build/klt /home/pervinco/Datasets/KITTI/dataset/sequences/00/image_0 100

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <filesystem>

int main(int argc, char **argv) {
    if (argc != 3) 
    {
        std::cerr << "image_path num_frames " << std::endl;
        return 0;
    }

    auto image_path = std::filesystem::path(argv[1]);
    const int num_frames = std::atoi(argv[2]);

    // image_path에 있는 파일들의 이름을 벡터에 저장
    std::vector<std::string> image_filenames;
    image_filenames.reserve(5000); // 벡터의 크기를 미리 5000으로 할당
    
    // 벡터에 파일 이름들을 담는다.
    for (const auto &entry : std::filesystem::directory_iterator(image_path))
    {
        image_filenames.push_back(entry.path());
    }
    std::sort(image_filenames.begin(), image_filenames.end()); // 이름 순으로 정렬.
    image_filenames.resize(num_frames); // 크기를 num_frames로 조정.

    cv::Mat img, img_next; // k번째 이미지와 k+1번째 이미지
    std::vector<cv::Point2f> kpts, kpts_next; // k번째 특징점과 k+1번째 특징점

    // 종료 조건 설정
    // 특징점 추척 알고리즘이 언제 종료될지를 설정함. 최대 반복 횟수와 원하는 정확도에 기반하여 설정된다.
    // 최대 반복 횟수는 10, 원하는 정확도는 0.03으로 설정된다.
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
    std::vector<uchar> status; // 특징점 추적 상태 저장.
    std::vector<float> err; // 오차 저장.

    for (int i = 0; i < num_frames-1; i++)
    {
        img = cv::imread(image_filenames[i], cv::IMREAD_GRAYSCALE);
        img_next = cv::imread(image_filenames[i+1], cv::IMREAD_GRAYSCALE);

        cv::Mat img_vis = img_next.clone(); // 추적 결과를 시각화하기 위한 이미지 복사
        cv::cvtColor(img_vis, img_vis, cv::COLOR_GRAY2BGR); // 이미지를 3채널로 변환    

        // // k번째 이미지에 대해 tracking하고 있는 특징점이 50개 미만이면 특징점을 추출한다.
        // if (kpts.size() < 50)
        // {
        //     // shi-tomasi 알고리즘을 사용하여 특징점을 추출한다.
        //     // 현재 프레임에서 추적할 특징점의 좌표를 담고 있는 벡터를 kpts에 저장.
        //     cv::goodFeaturesToTrack(img, kpts, 1000, 0.01, 15);
        //     /* 
        //     현재의 방식은 tracking되고 있는 것들도 완전히 지우고 새롭게 keypoint를 찾아내고 tracking을 수행한다. 결과적으로 tracking이 제대로 이어지지 않는다. 
        //     goodFeaturesToTrack을 별도의 컨테이너에 담고 기존 kpts에 push_back하는 방식으로 변경해야 한다.
        //     그렇게 되면 기존 특징점들이 완전히 지워지지 않고 유지되면서 새로운 특징점들이 추가되어 추적이 이어질 수 있다.
        //     */
        // }

        // k번째 이미지에 대해 tracking하고 있는 특징점이 50개 미만이면 특징점을 추출한다.
        if (kpts.size() < 50)
        {
            std::vector<cv::Point2f> new_kpts;
            cv::goodFeaturesToTrack(img, new_kpts, 1000, 0.01, 15);
            kpts.insert(kpts.end(), new_kpts.begin(), new_kpts.end());
        }

        // KLT optical flow
        // k번째 이미지, k+1번째 이미지, k번째 특징점, k+1번째 특징점(아직 구하지 않았음), 특징점 추적 상태, 오차, 피라미드 레벨, 최대 반복 횟수, 종료 조건
        // k번째 이미지의 특징점으로 optical flow를 수행해서 추적된 특징점의 좌표를 저장할 벡터.
        // status는 i번째 특징점이 추적되었는지 여부를 저장하는 벡터.
        // err는 i번째 특징점의 추적 오차를 저장하는 벡터.  
        // cv::Size(21,21)는 해당 특징점을 어느 범위까지 추적할지 윈도우(range)를 설정.
        // 2는 피라미드 레벨을 설정.
        // criteria는 종료 조건을 설정.
        cv::calcOpticalFlowPyrLK(img, img_next, kpts, kpts_next, status, err, cv::Size(21,21), 2, criteria);

        // 시각화
        // k+1번째 이미지에 추적된 특징점을 시각화하기 위해 원을 그린다.
        for (size_t j = 0; j < kpts_next.size(); j++)
        {
            if (status[j])
            {
                cv::circle(img_vis, kpts_next[j], 4, cv::Scalar(0, 0, 255));
                // cv::line(img_vis, kpts[j], kpts_next[j], cv::Scalar(0, 255, 0));
                // 반대 방향으로 직선 그리기
                cv::Point2f direction = kpts[j] - kpts_next[j]; // 이동 방향 계산
                cv::Point2f opposite_end = kpts_next[j] - direction; // 반대 방향으로 연장된 점
                cv::line(img_vis, kpts_next[j], opposite_end, cv::Scalar(0, 255, 0), 1);
            }
        }

        cv::imshow("img_vis", img_vis);
        cv::waitKey(33);

        // optical flow 실패하거나 이미지 밖으로 나간 특징점들을 제거.
        int indexCorrection = 0;
        for (int j = 0; j < status.size(); j++)
        {
            cv::Point2f pt = kpts_next.at(j - indexCorrection);
            if ((status.at(j) == 0) || (pt.x < 0) || (pt.y < 0))
            {
                if ((pt.x < 0) || (pt.y < 0))
                {
                    status.at(j) = 0;
                }
                kpts.erase(kpts.begin() + (j - indexCorrection));
                kpts_next.erase(kpts_next.begin() + (j - indexCorrection));
                indexCorrection++;
            }
        }

        kpts = kpts_next; // k번째 특징점을 k+1번째 특징점으로 업데이트. 
        kpts_next.clear(); // k+1번째 특징점 벡터를 비운다.
        status.clear(); // 특징점 추적 상태 벡터를 비운다.
        err.clear(); // 오차 벡터를 비운다.
    }
}