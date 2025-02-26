
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

/*
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;
*/

#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

using Mat = cv::Mat;
using string = std::string;


const int nDelayTimes = 2;
string sData_path = "/root/lwx_dataset/EuRoC/V1_03_difficult/mav0/";
string sConfig_path = sData_path + "cam0/data.csv";


int image_width = 752, image_height = 480;

// ds cam0
double fx0 = 349.7560023050409, fy0 = 348.72454229977037, cx0 = 365.89440762590149, cy0 = 249.32995565708704, xi0 = -0.2409573942178872, alpha0 = 0.566996899163044;

// ds cam1
double fx1 = 361.6713883800533, fy1 = 360.5856493689301, cx1 = 379.40818394080869, cy1 = 255.9772968522045, xi1 = -0.21300835384809328, alpha1 = 0.5767008625037023;

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

// R_cam1_cam0
cv::Mat extrinsic_R = ( cv::Mat_<double> ( 3,3 ) <<  0.9999972564779, 0.002312067192424, 0.0003760081024156,
                                                    -0.002317135723281, 0.9998980485066, 0.01408983584665,
                                                    -0.0003433931205242, -0.01409066845271, 0.9999006626377);


cv::Mat extrinsic_t = ( cv::Mat_<double> ( 3,1 ) << -0.1100738081272,
                                                    0.0003991215470141,
                                                    -0.000853702503358 );

const cv::Mat K_left = ( cv::Mat_<double> ( 3,3 ) << _fx0,   0.0,                 _cx0, 
                                                  0.0,                   _fy0, _cy0, 
                                                  0.0,                   0.0,                 1.0 );

const cv::Mat D_left = ( cv::Mat_<double> ( 5,1 ) << k1, k2, p1, p2, 0.000000);


const cv::Mat K_right = ( cv::Mat_<double> ( 3,3 ) << _fx1,   0.0,                 _cx1, 
                                                0.0,                   _fy1, _cy1, 
                                                0.0,                   0.0,                 1.0 );

const cv::Mat D_right = ( cv::Mat_<double> ( 5,1 ) << k1, k2, p1, p2, 0.000000);




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
        }
    }


}


// 双球模型去畸变函数
Mat undistortDoubleSphere(const Mat& src, 
                         double fx, double fy,
                         double cx, double cy,
                         double xi, double alpha) {
    Mat dst = Mat::zeros(src.size(), src.type());
    Mat map_x(src.size(), CV_32F);
    Mat map_y(src.size(), CV_32F);

    for (int v = 0; v < src.rows; ++v) {
        for (int u = 0; u < src.cols; ++u) {
            // 逆投影计算
            double mx = (u - cx) / fx;
            double my = (v - cy) / fy;
            double r2 = mx*mx + my*my;

            // 有效性检查
            if (alpha > 0.5 && r2 > 1.0/(2*alpha - 1)) continue;

            // 计算mz
            double sqrt_term = sqrt(1 - (2*alpha - 1)*r2);
            double denominator = alpha*sqrt_term + (1 - alpha);
            if (fabs(denominator) < 1e-6) continue;
            double mz = (1 - alpha*alpha*r2) / denominator;

            // 3D方向向量
            double numerator = mz*xi + sqrt(mz*mz + (1 - xi*xi)*r2);
            double denominator_total = mz*mz + r2;
            if (fabs(denominator_total) < 1e-6) continue;
            double scale = numerator / denominator_total;

            double X = scale * mx;
            double Y = scale * my;
            double Z = scale * mz - xi;

            // 归一化
            double norm = sqrt(X*X + Y*Y + Z*Z);
            if (norm < 1e-6) continue;
            X /= norm; Y /= norm; Z /= norm;

            // 针孔投影
            if (Z <= 1e-6) continue;
            double u_src = fx * (X/Z) + cx;
            double v_src = fy * (Y/Z) + cy;

            map_x.at<float>(v, u) = u_src;
            map_y.at<float>(v, u) = v_src;
        }
    }

    remap(src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT);
    return dst;
}

// 双球模型去畸变核心函数（严格遵循公式40-45）
Mat undistortDoubleSphereCorrected(const Mat& distorted_img, 
                                  double fx, double fy, 
                                  double cx, double cy,
                                  double xi, 
                                  double alpha,
                                  double pinhole_fx, double pinhole_fy, 
                                  double pinhole_cx, double pinhole_cy) {
    Mat undistorted_img = Mat::zeros(distorted_img.size(), distorted_img.type());
    Mat map_x(distorted_img.size(), CV_32F);
    Mat map_y(distorted_img.size(), CV_32F);

    const int height = distorted_img.rows;
    const int width = distorted_img.cols;

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            // ------------------------------
            // Step 1: 计算无畸变坐标系的3D射线（针孔模型）
            // ------------------------------
            double x = (u - pinhole_cx) / pinhole_fx;  // 归一化x坐标
            double y = (v - pinhole_cy) / pinhole_fy;  // 归一化y坐标
            double z = 1.0;            // 假设深度为1

            // ------------------------------
            // Step 2: 应用双球模型正向投影（公式40-45）
            // ------------------------------
            // 公式41: 计算d1
            double d1 = sqrt(x*x + y*y + z*z);

            // 公式43-45: 检查有效投影区域
            double w1 = (alpha <= 0.5) ? (alpha/(1-alpha)) : ((1-alpha)/alpha);
            double w2 = (w1 + xi)/sqrt(2*w1*xi + xi*xi + 1);
            if (z <= -w2*d1) {
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }

            // 公式42: 计算d2
            double term_z = xi*d1 + z;
            double d2 = sqrt(x*x + y*y + term_z*term_z);

            // 公式40: 计算畸变坐标
            double denominator = alpha*d2 + (1-alpha)*term_z;
            if (fabs(denominator) < 1e-9) {
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }

            double u_dist = fx*(x/denominator) + cx;
            double v_dist = fy*(y/denominator) + cy;

            // ------------------------------
            // Step 3: 建立映射关系
            // ------------------------------
            // 注意这里映射方向是 undistorted -> distorted
            map_x.at<float>(v, u) = static_cast<float>(u_dist);
            map_y.at<float>(v, u) = static_cast<float>(v_dist);
        }
    }

    // ------------------------------
    // Step 4: 执行逆向重映射
    // ------------------------------
    remap(distorted_img, 
          undistorted_img, 
          map_x, 
          map_y, 
          INTER_LINEAR, 
          BORDER_CONSTANT, 
          Scalar(0, 0, 0));

    return undistorted_img;
}

int main2(const Mat& img0, const Mat& img1) {
    // 读取原始双目图像
    // Mat left_raw = imread("left_raw.jpg");
    // Mat right_raw = imread("right_raw.jpg");

    // 双球模型参数（需校准获取）
    const Size image_size(image_width, image_height);
    // 左相机参数
    const double fxL = 823.5, fyL = 823.2, cxL = 635.8, cyL = 382.4;
    const double xiL = 0.32, alphaL = 0.76;
    // 右相机参数
    const double fxR = 825.1, fyR = 824.8, cxR = 642.3, cyR = 378.9;
    const double xiR = 0.31, alphaR = 0.75;

    // Step 1: 双球模型去畸变
    // Mat left_undistorted = undistortDoubleSphere(left_raw, fxL, fyL, cxL, cyL, xiL, alphaL);
    // Mat right_undistorted = undistortDoubleSphere(right_raw, fxR, fyR, cxR, cyR, xiR, alphaR);

#if 0
    Mat left_undistorted = undistortDoubleSphereCorrected(img0, fx0, fy0, cx0, cy0, xi0, alpha0, fx0, fy0, cx0, cy0);
    Mat right_undistorted = undistortDoubleSphereCorrected(img1, fx1, fy1, cx1, cy1, xi1, alpha1, fx1, fy1, cx1, cy1);
#else
    Mat left_undistorted = undistortDoubleSphereCorrected(img0, fx0, fy0, cx0, cy0, xi0, alpha0, _fx0, _fy0, _cx0, _cy0);
    Mat right_undistorted = undistortDoubleSphereCorrected(img1, fx1, fy1, cx1, cy1, xi1, alpha1, _fx1, _fy1, _cx1, _cy1);
#endif

    // Step 2: 立体校正参数（需校准获取）
    // Mat R = (Mat_<double>(3,3) << 0.9999, 0.0008, -0.0112,
    //                              -0.0007, 1.0000, 0.0023,
    //                              0.0112, -0.0023, 0.9999);
    // Mat T = (Mat_<double>(3,1) << -120.34, 0.512, 1.234);

    Mat R = extrinsic_R;
    Mat T = extrinsic_t;

    // 构建针孔模型参数（畸变系数设为0）
#if 0
    // Mat cameraMatrixL = (Mat_<double>(3,3) << fxL, 0, cxL, 0, fyL, cyL, 0, 0, 1);
    // Mat cameraMatrixR = (Mat_<double>(3,3) << fxR, 0, cxR, 0, fyR, cyR, 0, 0, 1);
    Mat cameraMatrixL = (Mat_<double>(3,3) << fx0, 0, cx0, 0, fy0, cy0, 0, 0, 1);
    Mat cameraMatrixR = (Mat_<double>(3,3) << fx1, 0, cx1, 0, fy1, cy1, 0, 0, 1);
#else    
    Mat cameraMatrixL = K_left;
    Mat cameraMatrixR = K_right;
#endif

    Mat distCoeffsL = Mat::zeros(4, 1, CV_64F);
    Mat distCoeffsR = Mat::zeros(4, 1, CV_64F);

    // Step 3: 计算立体校正参数
    Mat R1, R2, P1, P2, Q;
    stereoRectify(cameraMatrixL, distCoeffsL,
                  cameraMatrixR, distCoeffsR,
                  image_size,
                  R, T,
                  R1, R2, P1, P2, Q/*,
                //   CALIB_ZERO_DISPARITY, 0, image_size);
                  CALIB_ZERO_DISPARITY, 0, image_size*/);

    // Step 4: 生成校正映射
    Mat rmapLx, rmapLy, rmapRx, rmapRy;
    initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1,
                           image_size, CV_32FC1, rmapLx, rmapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2,
                           image_size, CV_32FC1, rmapRx, rmapRy);

    // Step 5: 应用立体校正
    Mat left_rectified, right_rectified;
    remap(left_undistorted, left_rectified, rmapLx, rmapLy, INTER_LINEAR);
    remap(right_undistorted, right_rectified, rmapRx, rmapRy, INTER_LINEAR);

    // 可视化结果
    Mat canvas;
    hconcat(left_rectified, right_rectified, canvas);
    for (int i = 0; i < canvas.rows; i += 32)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1);

    imshow("Rectified", canvas);
    waitKey(0);

    return 0;
}

int main(int argc, char* argv[])
{

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

        filename = "1403715913534057984.png";
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
        main2(img0, img1);

        // cv::imshow("SOURCE IMAGE", img0);
		// cv::waitKey(1);

        // usleep(50000*nDelayTimes);
        // std::this_thread::sleep_for(std::chrono::milliseconds(50 * nDelayTimes));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
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