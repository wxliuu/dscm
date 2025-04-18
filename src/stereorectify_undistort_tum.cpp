
//@date: 2025-2-24
//@author: wxliu
//@brief: stereo rectify and undistort

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
// #include <chrono>
#include <thread>
#include <filesystem>
#include <regex>

#include <opencv2/opencv.hpp>
/*#include <cmath>
#include <vector>


using namespace std;
*/

using namespace cv;

using Mat = cv::Mat;
using string = std::string;


const int nDelayTimes = 2;
string sData_path = "/root/lwx_dataset/EuRoC/V1_03_difficult/mav0/";
string sConfig_path = sData_path + "cam0/data.csv";

#define _USE_THE_SAME_INTRINSICS_


int image_width = 512, image_height = 512;

// ds cam0
double fx0 = 158.28600034966977, fy0 = 158.2743455478755, cx0 = 254.96116578191653, cy0 = 256.8894394501779, xi0 = -0.17213086034353243, alpha0 = 0.5931177593944744;

// ds cam1
double fx1 = 157.91830144176309, fy1 = 157.8901286125632, cx1 = 252.56547609702953, cy1 = 255.02489416194656, xi1 = -0.17114780716007858, alpha1 = 0.5925543396658507;

// pinhole cam0
// intrinsics: [458.654, 457.296, 367.215, 248.375] #fu, fv, cu, cv
double _fx0 = 458.654, _fy0 = 457.296, _cx0 = 367.215, _cy0 = 248.375;

// distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05] #k1, k2, p1, p2

// pinhole cam1
// intrinsics: [457.587, 456.134, 379.999, 255.238] #fu, fv, cu, cv
double _fx1 = 457.587, _fy1 = 456.134, _cx1 = 379.999, _cy1 = 255.238;

// distortion_coefficients: [-0.28368365,  0.07451284, -0.00010473, -3.55590700e-05]

// radial-tangential
double k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;

/*
T_c1_c0=
 0.9999995881755 -0.0008429104329 -0.0003363788793 -0.1009678603465
 0.0008258897264  0.9988593352731 -0.0477424993476 -0.0019668632282
 0.0003762378345  0.0477422018742  0.9988596200695 -0.0015905749062
 0.0000000000000  0.0000000000000  0.0000000000000  1.0000000000000
*/ 
// R_cam1_cam0
cv::Mat extrinsic_R = ( cv::Mat_<double> ( 3,3 ) << 0.9999995881755, -0.0008429104329, -0.0003363788793,
                                                    0.0008258897264, 0.9988593352731, -0.0477424993476,
                                                    0.0003762378345, 0.0477422018742, 0.9988596200695);


cv::Mat extrinsic_t = ( cv::Mat_<double> ( 3,1 ) << -0.1009678603465,
                                                    -0.0019668632282,
                                                    -0.0015905749062 );

const cv::Mat K_left = ( cv::Mat_<double> ( 3,3 ) << _fx0,   0.0,                 _cx0, 
                                                  0.0,                   _fy0, _cy0, 
                                                  0.0,                   0.0,                 1.0 );

const cv::Mat D_left = ( cv::Mat_<double> ( 5,1 ) << k1, k2, p1, p2, 0.000000);


const cv::Mat K_right = ( cv::Mat_<double> ( 3,3 ) << _fx1,   0.0,                 _cx1, 
                                                0.0,                   _fy1, _cy1, 
                                                0.0,                   0.0,                 1.0 );

const cv::Mat D_right = ( cv::Mat_<double> ( 5,1 ) << k1, k2, p1, p2, 0.000000);

cv::Mat map_x_cam0_final, map_y_cam0_final, map_x_cam1_final, map_y_cam1_final;

void removeNewline(std::string& str) {
    if (!str.empty() && str.back() == '\n') {
        str.pop_back();  // 去掉 \n
    }
    if (!str.empty() && str.back() == '\r') {
        str.pop_back();  // 去掉 \r
    }
}

void stereoRectifyAndUndistort(const Mat& img0, const Mat& img1) {
    // 1. Read the camera calibration parameters
    // 2. Read the stereo calibration parameters
    // 3. Rectify the stereo images
    // 4. Undistort the stereo images
    // 5. Display the stereo images

    cv::Mat map(image_height, image_width, CV_32FC1);
    cv::Mat map1_L, map2_L, map1_R, map2_R;

    cv::Mat R1, R2, P1, P2, Q;

    cv::Mat map1x, map1y, map2x, map2y;
    cv::stereoRectify(K_left, D_left, K_right, D_right, cv::Size(image_width, image_height), extrinsic_R, extrinsic_t, R1, R2, P1, P2, Q);
    cv::initUndistortRectifyMap(K_left, D_left, R1, P1, cv::Size(image_width, image_height), CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(K_right, D_right, R2, P2, cv::Size(image_width, image_height), CV_32FC1, map2x, map2y);

    // img0
    Mat map_x(img0.size(), CV_32F);
    Mat map_y(img0.size(), CV_32F);

    const int height = img0.rows;
    const int width = img0.cols;
    // double fx = _fx0, fy = _fy0, cx = _cx0, cy = _cy0;
    // double fx = fx0, fy = fy0, cx = cx0, cy = cy0;
    // double xi = xi0, alpha = alpha0;

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            // ------------------------------
            // Step 1: 计算无畸变坐标系的3D射线（针孔模型）
            // ------------------------------
            double x = (u - _cx0) / _fx0;  // 归一化x坐标
            double y = (v - _cy0) / _fy0;  // 归一化y坐标
            double z = 1.0;            // 假设深度为1

            // ------------------------------
            // Step 2: 应用双球模型正向投影（公式40-45）
            // ------------------------------
            // 公式41: 计算d1
            double d1 = sqrt(x*x + y*y + z*z);

            // 公式43-45: 检查有效投影区域
            double w1 = (alpha0 <= 0.5) ? (alpha0/(1-alpha0)) : ((1-alpha0)/alpha0);
            double w2 = (w1 + xi0)/sqrt(2*w1*xi0 + xi0*xi0 + 1);
            if (z <= -w2*d1) {
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }

            // 公式42: 计算d2
            double term_z = xi0*d1 + z;
            double d2 = sqrt(x*x + y*y + term_z*term_z);

            // 公式40: 计算畸变坐标
            double denominator = alpha0*d2 + (1-alpha0)*term_z;
            if (fabs(denominator) < 1e-9) {
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }

            double u_dist = fx0*(x/denominator) + cx0;
            double v_dist = fy0*(y/denominator) + cy0;

            // ------------------------------
            // Step 3: 建立映射关系
            // ------------------------------
            // 注意这里映射方向是 undistorted -> distorted
            // map_x.at<float>(v, u) = static_cast<float>(u_dist);
            // map_y.at<float>(v, u) = static_cast<float>(v_dist);

            map_x.at<float>(v, u) = map1x.at<float>(v_dist, u_dist);
            map_y.at<float>(v, u) = map1y.at<float>(v_dist, u_dist);
        }
    }

    Mat left_rectified, right_rectified;
    cv::remap(img0, left_rectified, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));


    // img1
    {
    Mat map_x(img0.size(), CV_32F);
    Mat map_y(img0.size(), CV_32F);

    const int height = img0.rows;
    const int width = img0.cols;
    // double fx = _fx0, fy = _fy0, cx = _cx0, cy = _cy0;
    // double fx = fx0, fy = fy0, cx = cx0, cy = cy0;
    // double xi = xi0, alpha = alpha0;

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            // ------------------------------
            // Step 1: 计算无畸变坐标系的3D射线（针孔模型）
            // ------------------------------
            double x = (u - _cx1) / _fx1;  // 归一化x坐标
            double y = (v - _cy1) / _fy1;  // 归一化y坐标
            double z = 1.0;            // 假设深度为1

            // ------------------------------
            // Step 2: 应用双球模型正向投影（公式40-45）
            // ------------------------------
            // 公式41: 计算d1
            double d1 = sqrt(x*x + y*y + z*z);

            // 公式43-45: 检查有效投影区域
            double w1 = (alpha1 <= 0.5) ? (alpha1/(1-alpha1)) : ((1-alpha1)/alpha1);
            double w2 = (w1 + xi1)/sqrt(2*w1*xi1 + xi1*xi1 + 1);
            if (z <= -w2*d1) {
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }

            // 公式42: 计算d2
            double term_z = xi1*d1 + z;
            double d2 = sqrt(x*x + y*y + term_z*term_z);

            // 公式40: 计算畸变坐标
            double denominator = alpha1*d2 + (1-alpha1)*term_z;
            if (fabs(denominator) < 1e-9) {
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }

            double u_dist = fx1*(x/denominator) + cx1;
            double v_dist = fy1*(y/denominator) + cy1;

            // ------------------------------
            // Step 3: 建立映射关系
            // ------------------------------
            // 注意这里映射方向是 undistorted -> distorted
            // map_x.at<float>(v, u) = static_cast<float>(u_dist);
            // map_y.at<float>(v, u) = static_cast<float>(v_dist);

            map_x.at<float>(v, u) = map1x.at<float>(v_dist, u_dist);
            map_y.at<float>(v, u) = map1y.at<float>(v_dist, u_dist);
        }
    }

        cv::remap(img1, right_rectified, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    Mat canvas;
    hconcat(left_rectified, right_rectified, canvas);
    for (int i = 0; i < canvas.rows; i += 32)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1);

    imshow("Rectified", canvas);
    waitKey(0);

}

// improved version on 2025-2-26
void stereoRectifyAndUndistort() {
    // 1. Read the camera calibration parameters
    // 2. Read the stereo calibration parameters
    // 3. Rectify the stereo images
    // 4. Undistort the stereo images
    // 5. Display the stereo images


    cv::Mat R1, R2, P1, P2, Q;
    cv::Mat map_stereorectify_cam0_x, map_stereorectify_cam0_y, map_stereorectify_cam1_x, map_stereorectify_cam1_y;

#if defined(_USE_THE_SAME_INTRINSICS_)
    Mat cameraMatrixL = (Mat_<double>(3,3) << fx0, 0, cx0, 0, fy0, cy0, 0, 0, 1);
    Mat cameraMatrixR = (Mat_<double>(3,3) << fx1, 0, cx1, 0, fy1, cy1, 0, 0, 1);
#else // use respective intrinsic parameters
    Mat cameraMatrixL = K_left;
    Mat cameraMatrixR = K_right;
#endif

    cv::stereoRectify(cameraMatrixL, D_left, cameraMatrixR, D_right, cv::Size(image_width, image_height), extrinsic_R, extrinsic_t, R1, R2, P1, P2, Q);
    cv::initUndistortRectifyMap(cameraMatrixL, D_left, R1, P1, cv::Size(image_width, image_height), CV_32FC1, map_stereorectify_cam0_x, map_stereorectify_cam0_y);
    cv::initUndistortRectifyMap(cameraMatrixR, D_right, R2, P2, cv::Size(image_width, image_height), CV_32FC1, map_stereorectify_cam1_x, map_stereorectify_cam1_y);

    // img0
    // Mat map_x(img0.size(), CV_32F);
    // Mat map_y(img0.size(), CV_32F);

    Mat map_undistort_cam0_x(cv::Size(image_width, image_height), CV_32FC1);
    Mat map_undistort_cam0_y(cv::Size(image_width, image_height), CV_32FC1);

    Mat map_undistort_cam1_x(image_height, image_width, CV_32FC1);
    Mat map_undistort_cam1_y(image_height, image_width, CV_32FC1);

    for (int v = 0; v < image_height; ++v) {
        for (int u = 0; u < image_width; ++u) {
            // ------------------------------
            // Step 1: 计算无畸变坐标系的3D射线（针孔模型）
            // ------------------------------
            double x = (u - cameraMatrixL.at<double>(0, 2)) / cameraMatrixL.at<double>(0, 0);  // 归一化x坐标
            double y = (v - cameraMatrixL.at<double>(1, 2)) / cameraMatrixL.at<double>(1, 1);  // 归一化y坐标
            double z = 1.0;            // 假设深度为1

            // ------------------------------
            // Step 2: 应用双球模型正向投影（公式40-45）
            // ------------------------------
            // 公式41: 计算d1
            double d1 = sqrt(x*x + y*y + z*z);

            // 公式43-45: 检查有效投影区域
            double w1 = (alpha0 <= 0.5) ? (alpha0/(1-alpha0)) : ((1-alpha0)/alpha0);
            double w2 = (w1 + xi0)/sqrt(2*w1*xi0 + xi0*xi0 + 1);
            if (z <= -w2*d1) {
                map_undistort_cam0_x.at<float>(v, u) = -1;
                map_undistort_cam0_y.at<float>(v, u) = -1;
                continue;
            }

            // 公式42: 计算d2
            double term_z = xi0*d1 + z;
            double d2 = sqrt(x*x + y*y + term_z*term_z);

            // 公式40: 计算畸变坐标
            double denominator = alpha0*d2 + (1-alpha0)*term_z;
            if (fabs(denominator) < 1e-9) {
                map_undistort_cam0_x.at<float>(v, u) = -1;
                map_undistort_cam0_y.at<float>(v, u) = -1;
                continue;
            }

            double u_dist = fx0*(x/denominator) + cx0;
            double v_dist = fy0*(y/denominator) + cy0;

            // ------------------------------
            // Step 3: 建立映射关系
            // ------------------------------
            // 注意这里映射方向是 undistorted -> distorted
            map_undistort_cam0_x.at<float>(v, u) = static_cast<float>(u_dist);
            map_undistort_cam0_y.at<float>(v, u) = static_cast<float>(v_dist);

        }
    }

    // Mat left_rectified, right_rectified;
    // cv::remap(img0, left_rectified, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));


    // img1
    for (int v = 0; v < image_height; ++v) {
        for (int u = 0; u < image_width; ++u) {
            // ------------------------------
            // Step 1: 计算无畸变坐标系的3D射线（针孔模型）
            // ------------------------------
            double x = (u - cameraMatrixR.at<double>(0, 2)) / cameraMatrixR.at<double>(0, 0);  // 归一化x坐标
            double y = (v - cameraMatrixR.at<double>(1, 2)) / cameraMatrixR.at<double>(1, 1);  // 归一化y坐标
            double z = 1.0;            // 假设深度为1

            // ------------------------------
            // Step 2: 应用双球模型正向投影（公式40-45）
            // ------------------------------
            // 公式41: 计算d1
            double d1 = sqrt(x*x + y*y + z*z);

            // 公式43-45: 检查有效投影区域
            double w1 = (alpha1 <= 0.5) ? (alpha1/(1-alpha1)) : ((1-alpha1)/alpha1);
            double w2 = (w1 + xi1)/sqrt(2*w1*xi1 + xi1*xi1 + 1);
            if (z <= -w2*d1) {
                map_undistort_cam1_x.at<float>(v, u) = -1;
                map_undistort_cam1_y.at<float>(v, u) = -1;
                continue;
            }

            // 公式42: 计算d2
            double term_z = xi1*d1 + z;
            double d2 = sqrt(x*x + y*y + term_z*term_z);

            // 公式40: 计算畸变坐标
            double denominator = alpha1*d2 + (1-alpha1)*term_z;
            if (fabs(denominator) < 1e-9) {
                map_undistort_cam1_x.at<float>(v, u) = -1;
                map_undistort_cam1_y.at<float>(v, u) = -1;
                continue;
            }

            double u_dist = fx1*(x/denominator) + cx1;
            double v_dist = fy1*(y/denominator) + cy1;

            // ------------------------------
            // Step 3: 建立映射关系
            // ------------------------------
            // 注意这里映射方向是 undistorted -> distorted
            map_undistort_cam1_x.at<float>(v, u) = static_cast<float>(u_dist);
            map_undistort_cam1_y.at<float>(v, u) = static_cast<float>(v_dist);

        }
    }

    // cv::remap(img1, right_rectified, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::remap(map_undistort_cam0_x, map_x_cam0_final, map_stereorectify_cam0_x, map_stereorectify_cam0_y, cv::INTER_CUBIC);
    cv::remap(map_undistort_cam0_y, map_y_cam0_final, map_stereorectify_cam0_x, map_stereorectify_cam0_y, cv::INTER_CUBIC);
    cv::remap(map_undistort_cam1_x, map_x_cam1_final, map_stereorectify_cam1_x, map_stereorectify_cam1_y, cv::INTER_CUBIC);
    cv::remap(map_undistort_cam1_y, map_y_cam1_final, map_stereorectify_cam1_x, map_stereorectify_cam1_y, cv::INTER_CUBIC);

    // Mat canvas;
    // hconcat(left_rectified, right_rectified, canvas);
    // for (int i = 0; i < canvas.rows; i += 32)
    //     line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1);

    // imshow("Rectified", canvas);
    // waitKey(0);

}

int main(int argc, char* argv[])
{
#if 1 
    string image0Path = "/root/lwx_dataset/tum_test/1520530424604726258_cam0.png";
    string image1Path = "/root/lwx_dataset/tum_test/1520530424604726258_cam1.png";

    cv::Mat img0 = cv::imread(image0Path.c_str(), cv::IMREAD_GRAYSCALE);
    cv::Mat img1 = cv::imread(image1Path.c_str(), cv::IMREAD_GRAYSCALE);
    if (img0.empty() || img1.empty())
    {
        std::cerr << "image is empty! \npath=" << image0Path << std::endl;
        return -1;
    }

    stereoRectifyAndUndistort();

    Mat left_rectified, right_rectified;
    left_rectified = Mat::zeros(img0.size(), img0.type());
    right_rectified = Mat::zeros(img1.size(), img1.type()); // 这样对齐后，多余的部分应该就是黑色的区域

    cv::remap(img0, left_rectified, map_x_cam0_final, map_y_cam0_final, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::remap(img1, right_rectified, map_x_cam1_final, map_y_cam1_final, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    // cv::remap(img0, left_rectified, map_x_cam0_final, map_y_cam0_final, cv::INTER_LINEAR);
    // cv::remap(img1, right_rectified, map_x_cam1_final, map_y_cam1_final, cv::INTER_LINEAR);

    // 可视化结果
    Mat canvas;
    hconcat(left_rectified, right_rectified, canvas);
    for (int i = 0; i < canvas.rows; i += 32)
        cv::line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1);

    imshow("Rectified", canvas);
    waitKey(0);

    // cv::imshow("SOURCE IMAGE", img0);
    // cv::waitKey(1);

    // save rectified images
    cv::imwrite("/root/lwx_dataset/tum_test/1520530424604726258_cam0_rectified.png", left_rectified);
    cv::imwrite("/root/lwx_dataset/tum_test/1520530424604726258_cam1_rectified.png", right_rectified);
    return 0;
#endif

#if 0    
    // /root/lwx_dataset/EuRoC/V1_03_difficult/mav0/cam0/data/1403715886584058112.png
    // 
    // Mat src = cv::imread("/root/lwx_dataset/EuRoC/V1_03_difficult/mav0/cam0/data/1403715886584058112.png", cv::IMREAD_GRAYSCALE);
    string filename = "1403715886584058112.png";
    string imagePath = "/root/lwx_dataset/EuRoC/V1_03_difficult/mav0/cam0/data/" + filename;
    Mat src = cv::imread(imagePath.c_str(), cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: could not load image!" << std::endl;
        return -1;
    }
    else {
        std::cout << "load image success!" << std::endl;

        cv::imshow("SOURCE IMAGE", src);
		cv::waitKey(0);

        return 0;
    } 
#endif

    stereoRectifyAndUndistort();

#if 1  
    // 打开文件
    // std::ifstream file("data.csv");
    std::ifstream file(sConfig_path);
    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }

    std::string line;
    // 跳过第一行（标题行）
    std::getline(file, line);

    // 循环读取每一行
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string timestamp, filename;

        // 读取时间戳和文件名
        if (std::getline(iss, timestamp, ',') && std::getline(iss, filename)) {
            // 输出时间戳和文件名
            std::cout << "时间戳: " << timestamp << ", 文件名: " << filename << std::endl;
        } else {
            std::cerr << "解析行失败: " << line << std::endl;
        }
        // std::cout << "filename.length() = " << filename.length() << std::endl;
        // 通过打印字符串长度才发现字符末尾存在回车符和换行符的情况: 0x0d（回车符，\r）和 0x0a（换行符，\n，导致文件名读取失败
        // filename = filename.substr(0, filename.length() - 1); // str.substr(pos, len)：从 pos 位置开始截取长度为 len 的子字符串。
        // filename.erase(filename.find_last_not_of("\r\n") + 1);
        filename = std::regex_replace(filename, std::regex("[\r\n]+$"), "");

        // filename = "1403715913534057984.png";

        // read image0
        string image0Path = sData_path + "cam0/data/" + filename;
        // Check if the file exists before attempting to read it
        if (!std::filesystem::exists(image0Path)) {
            std::cerr << "文件不存在: " << image0Path << std::endl;
            // continue; // Skip to the next image
            return -1;
        }
        

		cv::Mat img0 = cv::imread(image0Path.c_str(), cv::IMREAD_GRAYSCALE);
		if (img0.empty())
		{
			std::cerr << "image is empty! \npath=" << image0Path << std::endl;
			return -1;
		}

        // read image1
        string image1Path = sData_path + "cam1/data/" + filename;
        // Check if the file exists before attempting to read it
        if (!std::filesystem::exists(image1Path)) {
            std::cerr << "文件不存在: " << image1Path << std::endl;
            // continue; // Skip to the next image
            return -1;
        }
        

		cv::Mat img1 = cv::imread(image1Path.c_str(), cv::IMREAD_GRAYSCALE);
		if (img1.empty())
		{
			std::cerr << "image is empty! \npath=" << image1Path << std::endl;
			return -1;
		}

        // stereoRectifyAndUndistort(img0, img1);

        Mat left_rectified, right_rectified;
        left_rectified = Mat::zeros(img0.size(), img0.type());
        right_rectified = Mat::zeros(img1.size(), img1.type()); // 这样对齐后，多余的部分应该就是黑色的区域

        cv::remap(img0, left_rectified, map_x_cam0_final, map_y_cam0_final, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::remap(img1, right_rectified, map_x_cam1_final, map_y_cam1_final, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        // cv::remap(img0, left_rectified, map_x_cam0_final, map_y_cam0_final, cv::INTER_LINEAR);
        // cv::remap(img1, right_rectified, map_x_cam1_final, map_y_cam1_final, cv::INTER_LINEAR);

        // 可视化结果
        Mat canvas;
        hconcat(left_rectified, right_rectified, canvas);
        for (int i = 0; i < canvas.rows; i += 32)
            cv::line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1);

        imshow("Rectified", canvas);
        waitKey(1);

        // cv::imshow("SOURCE IMAGE", img0);
		// cv::waitKey(1);

        // save rectified images
        cv::imwrite("/root/lwx_dataset/EuRoC/V1_03_difficult_rectified/cam0/" + filename, left_rectified);
        cv::imwrite("/root/lwx_dataset/EuRoC/V1_03_difficult_rectified/cam1/" + filename, right_rectified);

        // usleep(50000*nDelayTimes);
        // std::this_thread::sleep_for(std::chrono::milliseconds(50 * nDelayTimes));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // 关闭文件
    file.close();
#else

    string sImage_file = sConfig_path;

	std::cout << "1 PubImageData start sImage_file: " << sImage_file << std::endl;

	std::ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open())
	{
		std::cerr << "Failed to open image file! " << sImage_file << std::endl;
		return -1;
	}

	std::string sImage_line;
	double dStampNSec;
	std::string sImgFileName;

    std::getline(fsImage, sImage_line); // skip the first line
	
	// cv::namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
	while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
	{
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec >> sImgFileName;
		std::cout << "Image t : " << std::fixed << dStampNSec << " Name: " << sImgFileName << std::endl;
		string imagePath = sData_path + "cam0/data/" + sImgFileName;

		Mat img = cv::imread(imagePath.c_str(), 0);
		if (img.empty())
		{
			std::cerr << "image is empty! path: " << imagePath << std::endl;
			return -1;
		}

		// cv::imshow("SOURCE IMAGE", img);
		// cv::waitKey(0);
		usleep(50000*nDelayTimes);
	}
	fsImage.close();

#endif

    return 0;
}