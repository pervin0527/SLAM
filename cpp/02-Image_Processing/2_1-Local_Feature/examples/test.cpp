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

vector<string> load_image_filenames(const filesystem::path &image_path, int num_frames)
{
    vector<string> image_filenames;
    image_filenames.reserve(5000); // 벡터의 크기를 미리 5000으로 할당.

    // 경로 내부에 있는 파일들을 모두 읽어와서 하나씩 벡터에 저장.
    for (const auto &entry : filesystem::directory_iterator(image_path))
    {
        image_filenames.push_back(entry.path().string());
    }

    sort(image_filenames.begin(), image_filenames.end()); // 이름 순으로 정렬.
    image_filenames.resize(num_frames); // 지정된 프레임 수만큼만 파일들을 유지.

    return image_filenames;
}

void processImages(const vector<string>& left_image_filenames, 
                   const vector<string>& right_image_filenames, 
                   int num_frames, 
                   cv::Ptr<cv::Feature2D> feature_detector, 
                   cv::Ptr<cv::BFMatcher> bf_matcher, 
                   cv::FlannBasedMatcher& knn_matcher) 
{
    cv::Mat img_left, img_right;
    cv::Mat desc_left, desc_right;
    vector<cv::KeyPoint> kpts_left, kpts_right;
    vector<cv::DMatch> bf_matches;
    vector<vector<cv::DMatch>> knn_matches;

    for(int i = 0; i < num_frames; i++) 
    {
        img_left = cv::imread(left_image_filenames[i], cv::IMREAD_GRAYSCALE);
        img_right = cv::imread(right_image_filenames[i], cv::IMREAD_GRAYSCALE);

        feature_detector->detectAndCompute(img_left, cv::Mat(), kpts_left, desc_left);
        feature_detector->detectAndCompute(img_right, cv::Mat(), kpts_right, desc_right);

        if(desc_left.empty() || desc_right.empty())
        {
            continue;
        }

        bf_matcher->match(desc_left, desc_right, bf_matches);
        std::vector<cv::DMatch> good_bf_matches;
        for (const auto &match : bf_matches)
        {
            if (match.distance < 50)
            {
                good_bf_matches.push_back(match);
            }
        }

        knn_matcher.knnMatch(desc_left, desc_right, knn_matches, 2);

        constexpr auto ratio_thresh = 0.8;
        std::vector<cv::DMatch> good_knn_matches;
        for (const auto &match : knn_matches)
        {
            if (match[0].distance < ratio_thresh * match[1].distance)
            {
                good_knn_matches.push_back(match[0]);
            }
        }

        cv::Mat img_bf;
        cv::drawMatches(img_left, kpts_left, img_right, kpts_right, good_bf_matches,
                        img_bf, cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        cv::Mat img_knn;
        cv::drawMatches(img_left, kpts_left, img_right, kpts_right,
                        good_knn_matches, img_knn, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        cv::imshow("BF Matches", img_bf);
        cv::imshow("KNN Matches", img_knn);
        cv::waitKey(0);
    }
}

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
    vector<string> left_image_filenames = load_image_filenames(left_image_path, num_frames);
    vector<string> right_image_filenames = load_image_filenames(right_image_path, num_frames);

    // ORB keypoint detector 정의. 최대 1000개의 특징점을 검출하도록 설정.
    auto feature_detector = cv::ORB::create(1000);

    // Matching 알고리즘(Brute-force, KNN) 정의.
    auto bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING); // Brute-force matcher 생성.
    auto knn_matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2)); // KNN matcher 생성.

    processImages(left_image_filenames, right_image_filenames, num_frames, feature_detector, bf_matcher, knn_matcher);
}