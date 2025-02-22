#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "left_image_path right_image_path num_frames " << std::endl;
    return 0;
  }

  // 파일 경로(왼쪽 카메라 이미지, 오른쪽 카메라 이미지)
  auto left_image_path = std::filesystem::path(argv[1]);
  auto right_image_path = std::filesystem::path(argv[2]);
  
  // 프레임 수
  const int num_frames = std::atoi(argv[3]);

  // 파일 이름 벡터
  std::vector<std::string> left_image_filenames, right_image_filenames;
  left_image_filenames.reserve(5000); // 벡터 예약 메모리 할당
  right_image_filenames.reserve(5000);

  // 왼쪽 카메라 이미지 파일 이름 벡터에 추가
  for (const auto &entry :
       std::filesystem::directory_iterator(left_image_path)) {
    left_image_filenames.push_back(entry.path());
  }

  // 오른쪽 카메라 이미지 파일 이름 벡터에 추가
  for (const auto &entry :
       std::filesystem::directory_iterator(right_image_path)) {
    right_image_filenames.push_back(entry.path());
  }

  // 파일 이름 벡터 정렬
  std::sort(left_image_filenames.begin(), left_image_filenames.end());
  std::sort(right_image_filenames.begin(), right_image_filenames.end());

  // 파일 이름 벡터 크기 조정
  left_image_filenames.resize(num_frames);
  right_image_filenames.resize(num_frames);

  /* Pangolin 윈도우 생성 및 3D 트래킹 플롯 생성 */
  // pangolin 윈도우 생성 및 3D 트래킹 플롯 생성
  pangolin::CreateWindowAndBind("Point cloud Viewer", 1024, 768); // 윈도우 생성
  glEnable(GL_DEPTH_TEST); // 깊이 테스트 활성화
  glEnable(GL_BLEND); // 블렌딩 활성화
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // 블렌딩 함수 설정

  // 카메라 설정
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1280, 720, 500, 500, 512, 389, 0.0001, 1000), // 투영 행렬
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)); // 모델 뷰 확인 함수

  // 핸들러 생성
  auto handler = std::make_unique<pangolin::Handler3D>(s_cam);

  // 디스플레이 생성
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, 0.0, 1.0, -1280.0f / 720.0f)
                              .SetHandler(handler.get());

  /* 이미지 처리 */
  cv::Mat img_left, img_right; // 이미지를 담을 행렬
  std::vector<cv::KeyPoint> kpts_left, kpts_right; // 키포인트를 담을 벡터
  cv::Mat desc_left, desc_right; // 디스크립터를 담을 행렬

  // ORB 특징 검출기 생성
  auto feature_detector = cv::ORB::create(1000);
  auto bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING); // brute-force 매칭기 생성
  std::vector<cv::DMatch> bf_matches; // 매칭 결과를 담을 벡터

  // 카메라 0과 카메라 1의 투영 행렬 (KITTI 캘리브레이션에서 가져옴)
  // projection matrix로 0번 카메라, 1번 카메라의 relative motion을 표현
  // clang-format off
    cv::Mat P0 = (cv::Mat_<float>(3, 4) <<
            7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00,
            0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);
    cv::Mat P1 = (cv::Mat_<float>(3, 4) <<
            7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02,
            0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00);
  // clang-format on

  int i = 0; // 프레임 인덱스
  while (!pangolin::ShouldQuit()) { // 윈도우가 닫히지 않았다면
    if (i == num_frames) { // 프레임 인덱스가 프레임 수와 같다면
      break; // 반복문 종료
    }

    // 이미지를 읽는다.
    img_left = cv::imread(left_image_filenames[i], cv::IMREAD_GRAYSCALE);
    img_right = cv::imread(right_image_filenames[i], cv::IMREAD_GRAYSCALE);

    // 특징 검출기를 사용하여 키포인트와 디스크립터를 추출
    feature_detector->detectAndCompute(img_left, cv::Mat(), kpts_left, desc_left);
    feature_detector->detectAndCompute(img_right, cv::Mat(), kpts_right, desc_right);

    // 디스크립터가 비어있다면 다음 프레임으로 넘어간다.
    if (desc_left.empty() || desc_right.empty()) {
      continue;
    }

    // brute-force 매칭
    bf_matcher->match(desc_left, desc_right, bf_matches);

    // 좋은 매칭 결과를 추출
    std::vector<cv::DMatch> good_bf_matches;
    for (const auto &match : bf_matches) {
      if (match.distance < 40) {
        good_bf_matches.push_back(match);
      }
    }

    // 브루투포스 매칭 결과를 이미지에 그린다.
    cv::Mat img_bf;
    cv::drawMatches(img_left, kpts_left, img_right, kpts_right, good_bf_matches,
                    img_bf, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 매칭 결과를 표시
    cv::imshow("BF Matches", img_bf);
    cv::waitKey(30);

    // cv::keypoint를 cv::Point2f로 변환.(triangulation이 Point2f를 사용하기 때문)
    std::vector<cv::Point2f> left_pts, right_pts;
    for (const auto &match : good_bf_matches) {
      left_pts.push_back(kpts_left[match.queryIdx].pt);
      right_pts.push_back(kpts_right[match.trainIdx].pt);
    }

    // 3D 트라이앵글레이션
    cv::Mat pts_4d;
    cv::triangulatePoints(P0, P1, left_pts, right_pts, pts_4d); // 출력값이 (wx, wy, wz, w)로 homogeneous coordinate이다.

    // 3D 트라이앵글레이션 결과를 시각화
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // 그래픽 버퍼 초기화
    d_cam.Activate(s_cam); // 카메라 활성화
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 배경색 설정
    glPointSize(3); // 포인트 크기 설정

    glBegin(GL_POINTS); // 3D 포인트 그리기 시작
    glColor3f(1.0, 0.0, 0.0); // 빨간색
    for (int i = 0; i < pts_4d.cols; i++) {
      cv::Mat x = pts_4d.col(i);
      x /= x.at<float>(3, 0); // homogeneous coordinate를 3차원 좌표로 변환
      glVertex3d(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2, 0)); // x, y, z 좌표를 그린다.
      //            std::cout << x.at<float>(0, 0) << " " << x.at<float>(1, 0)
      //            << " "
      //                      << x.at<float>(2, 0) << std::endl;
    }
    glEnd(); // 3D 포인트 그리기 종료

    pangolin::FinishFrame(); // 프레임 종료
    usleep(5000); // 5ms 대기

    // 프레임 인덱스 증가
    i++;
  }
}
