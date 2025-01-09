#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <filesystem> // C++17부터 추가됨. 파일을 읽을 때 사용.

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

int main(int args, char **argv)
{
    if (args != 4)
    {
        // 표준 에러 출력 스트림 cerr
        std::cerr << "Usage: left_image_path right_image_path num_frames" << std::endl;
        return 0;
    }

    auto left_image_path = std::filesystem::path(argv[1]);
    auto right_image_path = std::filesystem::path(argv[2]);
    const int num_frames = std::atoi(argv[3]); // 읽을 프레임 수. atoi는 문자열을 정수로 변환.

    std::vector<std::string> left_image_filenames, right_image_filenames;
    left_image_filenames.reserve(5000);
    right_image_filenames.reserve(5000);

    for (const auto &entry : std::filesystem::directory_iterator(left_image_path))
    {
        left_image_filenames.push_back(entry.path().string());
    }

    for (const auto &entry : std::filesystem::directory_iterator(right_image_path))
    {
        right_image_filenames.push_back(entry.path().string());
    }

    // 파일들을 이름 순서대로 정렬.
    std::sort(left_image_filenames.begin(), left_image_filenames.end());
    std::sort(right_image_filenames.begin(), right_image_filenames.end());

    // 지정된 프레임 수만큼만 파일들을 유지.
    left_image_filenames.resize(num_frames);
    right_image_filenames.resize(num_frames);

    // 이미지를 담을 컨테이너(비어 있는 행렬을 정의함.)
    cv::Mat img_left, img_right;
    cv::Mat desc_left, desc_right;                   // descriptor를 담을 행렬.
    std::vector<cv::KeyPoint> kpts_left, kpts_right; // keypoint를 담을 벡터.

    // ORB detector 생성.
    auto feature_detector = cv::ORB::create(1000);                                               // 이미지당 1000개의 ORB feature(keypoint)를 추출.
    auto bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);                                   // Brute-force matcher 생성.
    auto knn_matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2)); // K-Nearest Neighbors matcher 생성.

    std::vector<cv::DMatch> bf_matches;               // bf 매치 결과를 담을 벡터.
    std::vector<std::vector<cv::DMatch>> knn_matches; // knn 매치 결과를 담을 벡터. 1개에 대해 여러 개의 매칭 결과가 나오기 때문에 다차원 배열임.

    // 이미지 파일들을 읽어오는 반복문.
    for (int i = 0; i < num_frames; i++)
    {
        img_left = cv::imread(left_image_filenames[i], cv::IMREAD_GRAYSCALE);
        img_right = cv::imread(right_image_filenames[i], cv::IMREAD_GRAYSCALE);

        // cv::Mat()은 이미지의 특정 부분에 대해 detect를 하지 않기 위함인데 Mat()인 경우 전체 이미지에 대해 detect를 하는 것.
        // detect는 feature를 찾는 것이고 compute는 descriptor를 계산하는 것.
        // 계산된 key point는 kpts_left, kpts_right에 저장되고 descriptor는 desc_left, desc_right에 저장됨.
        feature_detector->detectAndCompute(img_left, cv::Mat(), kpts_left, desc_left);
        feature_detector->detectAndCompute(img_right, cv::Mat(), kpts_right, desc_right);

        // feature나 descriptor가 없는 경우 다음 프레임으로 넘어감.
        if (desc_left.empty() || desc_right.empty())
        {
            continue;
        }

        // 매칭 단계에서 중요한 점 --> 최대한 1대1 매칭이 되는 것이 좋다.
        // brute force matching
        bf_matcher->match(desc_left, desc_right, bf_matches);

        // 임계치 이상의 매칭 결과만 남긴다.
        std::vector<cv::DMatch> good_bf_matches; // 임계치 이상의 매칭을 저장할 벡터
        for (const auto &match : bf_matches)
        {
            if (match.distance < 50)
            {
                good_bf_matches.push_back(match);
            }
        }

        // KNN matching
        knn_matcher.knnMatch(desc_left, desc_right, knn_matches, 2); // 2순위까지 NN을 탐색. SNN을 하기 위함.

        // one-to-many matching을 회피하여 one-to-one matching을 하기 위한 SNN(Second Nearest Neighbor) 알고리즘
        constexpr auto ratio_thresh = 0.8;
        std::vector<cv::DMatch> good_knn_matches;
        for (const auto &match : knn_matches)
        {
            if (match[0].distance < ratio_thresh * match[1].distance)
            {
                good_knn_matches.push_back(match[0]);
            }
        }

        // 시각화
        // Draw Brute-force matches
        cv::Mat img_bf;
        cv::drawMatches(img_left, kpts_left, img_right, kpts_right, good_bf_matches,
                        img_bf, cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Draw KNN matches
        cv::Mat img_knn;
        cv::drawMatches(img_left, kpts_left, img_right, kpts_right,
                        good_knn_matches, img_knn, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Show matches
        cv::imshow("BF Matches", img_bf);
        cv::imshow("KNN Matches", img_knn);
        cv::waitKey(0);
    }

    return 0;
}
