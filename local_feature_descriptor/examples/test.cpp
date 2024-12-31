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
        cerr << "Usage: left_image_path right_image_path num_frames" << endl;
        return 0;
    }

    auto left_image_path = filesystem::path(argv[1]);
    auto right_image_path = filesystem::path(argv[2]);
    const int num_frames = atoi(argv[3]); // 읽을 프레임 수. atoi는 문자열을 정수로 변환.

    // 이미지 파일들을 포함하고 있는 디렉터리 경로.
    vector<string> left_image_filenames, right_image_filenames;
    left_image_filenames.reserve(5000);
    right_image_filenames.reserve(5000);

    // 경로 내부에 있는 파일들을 모두 읽어와서 하나씩 벡터에 저장.
    for(const auto &entry : filesystem::directory_iterator(left_image_path))
    {
        // cout << entry.path().string() << endl;
        left_image_filenames.push_back(entry.path().string());
    }

    for (const auto &entry : filesystem::directory_iterator(right_image_path))
    {
        // cout << entry.path().string() << endl;   
        right_image_filenames.push_back(entry.path().string());
    }

    // 이름 순으로 정렬.
    sort(left_image_filenames.begin(), left_image_filenames.end());
    sort(right_image_filenames.begin(), right_image_filenames.end());
    cout << left_image_filenames.size() << " " << right_image_filenames.size() << endl;

    // 지정된 프레임 수만큼만 파일들을 유지.
    left_image_filenames.resize(num_frames);
    right_image_filenames.resize(num_frames);    
    cout << left_image_filenames.size() << " " << right_image_filenames.size() << endl;

    cv::Mat img_left, img_right;
    cv::Mat desc_left, desc_right;
    vector<cv::KeyPoint> kpts_left, kpts_right;

    auto feature_detector = cv::ORB::create(1000); // ORB keypoint detector 생성. 최대 1000개의 특징점을 검출하도록 설정.
    auto bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING); // Brute-force matcher 생성.
    auto knn_matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2)); // KNN matcher 생성.

    vector<cv::DMatch> bf_matches;
    vector<vector<cv::DMatch>> knn_matches;
    for(int i = 0; i < num_frames; i++) // i번째 left, right 이미지를 각각 읽어 오는 방식.
    {
        img_left = cv::imread(left_image_filenames[i], cv::IMREAD_GRAYSCALE);
        img_right = cv::imread(right_image_filenames[i], cv::IMREAD_GRAYSCALE);

        // ORB detector로 keypoint와 descriptor를 검출.
        // 두번째 인자는 이미지의 특정 부분에 대해 detect를 하지 않기 위함인데 cv::Mat()으로 전달되는 경우 전체 이미지에 대해 detect를 하는 것.
        // 세번째, 네번째 인자는 검출된 keypoint와 descriptor를 저장하는 변수.
        feature_detector->detectAndCompute(img_left, cv::Mat(), kpts_left, desc_left);
        feature_detector->detectAndCompute(img_right, cv::Mat(), kpts_right, desc_right);

        cout << "Left Image Keypoints: " << kpts_left.size() << endl; // 1000
        cout << "Right Image Keypoints: " << kpts_right.size() << endl; // 1000

        if (!kpts_left.empty()) {
            float x = kpts_left[0].pt.x;
            float y = kpts_left[0].pt.y;
            cout << "First keypoint in left image: (" << x << ", " << y << ")" << endl;
        }

        cout << "Left Image Descriptors: " << desc_left.rows << " x " << desc_left.cols << endl; // 1000 x 32
        cout << "Right Image Descriptors: " << desc_right.rows << " x " << desc_right.cols << endl; // 1000 x 32

        // feature나 descriptor가 없는 경우 다음 프레임으로 넘어감.
        if(desc_left.empty() || desc_right.empty())
        {
            continue;
        }

        // i번째 left 이미지의 keypoint, descriptor를 i번째 right 이미지의 keypoint, descriptor와 매칭.
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

        // KNN 매칭
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
}