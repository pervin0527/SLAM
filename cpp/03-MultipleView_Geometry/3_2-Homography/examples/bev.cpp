/*
000000.png 파일을 읽어서 특정 영역을 추출한다.
추출한 영역을 이용해서 homography matrix를 추출한다.
homography matrix를 이용해서 추출한 영역을 이용해서 원본 이미지를 변환한다.
*/

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

int main() {
  cv::Mat img_front = cv::imread("000000.png", cv::IMREAD_COLOR);
  cv::Mat img_front_vis = img_front.clone(); // 원본 이미지 복사
  cv::Mat img_bev; // bev 이미지 저장

  cv::Point tl = {500, 200}; // top-left
  cv::Point tr = {600, 200}; // top-right
  cv::Point bl = {180, 330}; // bottom-left
  cv::Point br = {750, 330}; // bottom-right

  cv::circle(img_front_vis, tl, 5, cv::Scalar(0, 0, 255), 10); // 원본 이미지에 원 그리기
  cv::circle(img_front_vis, tr, 5, cv::Scalar(0, 0, 255), 10);
  cv::circle(img_front_vis, bl, 5, cv::Scalar(0, 0, 255), 10);
  cv::circle(img_front_vis, br, 5, cv::Scalar(0, 0, 255), 10);

  cv::imshow("img_front_vis", img_front_vis);
  cv::waitKey(0);

  // bev 이미지 상에서 4개의 점이 어디에 있는지 명시
  cv::Point target_tl = {0, 0}; // top-left
  cv::Point target_tr = {640, 0}; // top-right
  cv::Point target_bl = {0, 480}; // bottom-left
  cv::Point target_br = {640, 480}; // bottom-right

  std::vector<cv::Point2f> src_points = {tl, tr, bl, br}; // 원본 이미지 상의 4개의 점
  std::vector<cv::Point2f> dst_points = {target_tl, target_tr, target_bl, target_br}; // bev 이미지 상의 4개의 점

  cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points); // homography matrix 추출

  cv::warpPerspective(img_front, img_bev, M, cv::Size(640, 480)); // 원본 이미지를 bev 이미지로 변환(homography matrix 이용해서 warping)

  cv::imshow("img_bev", img_bev);
  cv::waitKey(0);
}