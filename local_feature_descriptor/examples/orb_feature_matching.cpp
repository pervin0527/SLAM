#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <filesystem>

// ./orb_feature_matching /home/pervinco/Datasets/KITTI/dataset/sequences/00/image_0 /home/pervinco/Datasets/KITTI/dataset/sequences/00/image_1 100

int main(int args, char **argv){
    if(args != 4){
        // 표준 에러 출력 스트림 cerr
        std::cerr << "left_image_path right_image_path num_frames " << std::endl;
        return 0;
    }

    auto left_image_path = std::filesystem::path(argv[1]);
    auto right_image_path = std::filesystem::path(argv[2]);
    const int num_frames = std::atoi(argv[3]); // 문자열을 정수로 변환.

    std::vector<std::string> left_image_filenames, right_image_filenames;
    left_image_filenames.reserve(5000);
    right_image_filenames.reserve(5000);

    for (const auto &entry :std::filesystem::directory_iterator(left_image_path))
    {
        std::cout << entry.path() << "\n";
        left_image_filenames.push_back(entry.path());
    }

    for (const auto &entry : std::filesystem::directory_iterator(right_image_path)) 
    {
        right_image_filenames.push_back(entry.path());
    }
}