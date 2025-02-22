#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

/*
left.png, right.png 파일을 읽어서 특징 검출기를 사용하여 키포인트와 디스크립터를 추출한다.
brute-force 매칭기를 사용하여 매칭 결과를 추출한다.
fundamental matrix, essential matrix를 추출한다.
E 또는 F 행렬을 분해해서 relative motion(R, t)을 구한다.
*/

int main() {
  cv::Mat img_left = cv::imread("left.png", cv::IMREAD_GRAYSCALE);
  cv::Mat img_right = cv::imread("right.png", cv::IMREAD_GRAYSCALE);

  // ORB 특징 검출기 생성 
  auto feature_detector = cv::ORB::create(1000);
  // brute-force 매칭기 생성
  auto bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
  std::vector<cv::DMatch> bf_matches;


  std::vector<cv::KeyPoint> kpts_left, kpts_right; // 키포인트 저장 벡터
  cv::Mat desc_left, desc_right; // 디스크립터 저장 행렬

  feature_detector->detectAndCompute(img_left, cv::noArray(), kpts_left, desc_left);
  feature_detector->detectAndCompute(img_right, cv::noArray(), kpts_right, desc_right);

  // brute-force 매칭 
  bf_matcher->match(desc_left, desc_right, bf_matches);

  // 좋은 매칭 결과를 추출
  std::vector<cv::DMatch> good_bf_matches;
  for (const auto &match : bf_matches) {
    if (match.distance < 40) { // 거리가 40보다 작은 매칭 결과를 추출
      good_bf_matches.push_back(match);
    }
  }

  // 브루투포스 매칭 결과를 이미지에 그린다.
  cv::Mat img_bf;
  cv::drawMatches(img_left, kpts_left, img_right, kpts_right, good_bf_matches,
                  img_bf, cv::Scalar::all(-1), cv::Scalar::all(-1),
                  std::vector<char>(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cv::imshow("BF Matches", img_bf);
  cv::waitKey(0);

  // 매칭 결과(cv::DMatch)를 cv::Point2f로 변환
  std::vector<cv::Point2f> pts_left, pts_right;
  for (int i = 0; i < bf_matches.size(); i++) {
    pts_left.push_back(kpts_left[bf_matches[i].queryIdx].pt);
    pts_right.push_back(kpts_right[bf_matches[i].trainIdx].pt);
  }

  // Fundamental matrix 추출
  // clang-format off
  // Intrinsic parameter
  cv::Mat K = (cv::Mat_<double>(3, 3)
               << 9.799200e+02, 0.000000e+00, 6.900000e+02,
                  0.000000e+00, 9.741183e+02, 2.486443e+02,
                  0.000000e+00, 0.000000e+00, 1.000000e+00);
  // clang-format on

  cv::Mat F = cv::findFundamentalMat(pts_left, pts_right, cv::FM_RANSAC, 3, 0.99);
  std::cout << "F = " << std::endl << F << std::endl;

  // Essential matrix 추출
  cv::Point2d pp(690, 248.6443); // 초점 좌표
  double focal = 979.92; // 초점 거리
  cv::Mat E = cv::findEssentialMat(pts_left, pts_right, focal, pp, cv::RANSAC, 0.999, 1.0);
  std::cout << "E = " << std::endl << E << std::endl;

  // R, t 구하기.
  cv::Mat R, t;
  cv::recoverPose(E, pts_left, pts_right, R, t, focal, pp);
  std::cout << "R = " << std::endl << R << std::endl;
  std::cout << "t = " << std::endl << t << std::endl;
}