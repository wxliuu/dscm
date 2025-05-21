
//@date: 2025-2-20
//@auther: wxliu
//@brief: DSCM: Double Sphere Camera Model

// undistort image: 1. ds unprojection 2. pinhole projection
// use the same fx.

// Multiple holes appeared in the image after processing.

// deepseek

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

// 双球模型去畸变核心算法
Mat undistortDoubleSphere(const Mat& distorted_img, 
                         double fx, double fy, 
                         double cx, double cy,
                         double xi, 
                         double alpha) {
    Mat undistorted_img = Mat::zeros(distorted_img.size(), distorted_img.type());
    Mat map_x(distorted_img.size(), CV_32F); 
    Mat map_y(distorted_img.size(), CV_32F);

    const int height = distorted_img.rows;
    const int width = distorted_img.cols;

    // 预计算去畸变映射关系
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            // ------------------------------
            // Step 1: 逆投影到归一化3D射线
            // ------------------------------
            // 公式(47)-(50)
            double mx = (u - cx) / fx;
            double my = (v - cy) / fy;
            double r2 = mx*mx + my*my;

            // 有效性检查（公式51）
            if (alpha > 0.5 && r2 > 1.0/(2*alpha - 1)) {
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }

            // 计算mz（公式50）
            double sqrt_term = sqrt(1 - (2*alpha - 1)*r2);
            double denominator = alpha*sqrt_term + (1 - alpha);
            if (fabs(denominator) < 1e-6) {
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }
            double mz = (1 - alpha*alpha*r2) / denominator;

            // 计算3D方向向量（公式46）
            double numerator = mz*xi + sqrt(mz*mz + (1 - xi*xi)*r2);
            double denominator_total = mz*mz + r2;
            if (fabs(denominator_total) < 1e-6) {
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }
            double scale = numerator / denominator_total;

            double X = scale * mx;
            double Y = scale * my;
            double Z = scale * mz - xi;

            // 归一化向量
            double norm = sqrt(X*X + Y*Y + Z*Z);
            if (norm < 1e-6) {
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }
            X /= norm;
            Y /= norm;
            Z /= norm;

            // ------------------------------
            // Step 2: 投影到无畸变针孔模型
            // ------------------------------
            if (Z <= 1e-6) { // 避免除以零
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }
            
            // 针孔投影公式
            double u_dst = fx * (X / Z) + cx;
            double v_dst = fy * (Y / Z) + cy;

            // 保存映射关系
            // map_x.at<float>(v, u) = static_cast<float>(u_dst);
            // map_y.at<float>(v, u) = static_cast<float>(v_dst);

            // modified on 2025-2-22
            if (u_dst >= 0 && v_dst >= 0 && u_dst < distorted_img.cols && v_dst < distorted_img.rows)
            {
                map_x.at<float>(v_dst, u_dst) = static_cast<float>(u);
                map_y.at<float>(v_dst, u_dst) = static_cast<float>(v);
            }
        }
    }

    // ------------------------------
    // Step 3: 执行重映射
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

// 示例使用
int main() {
    // 读取畸变图像
    // Mat distorted_img = imread("/home/lwx/dev/cpp_ws/dscm/assets/screen_20_double-sphere.jpg");
    Mat distorted_img = imread("../assets/screen_20_double-sphere.jpg");
    if (distorted_img.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return -1;
    }

    // 双球模型参数（需根据标定结果填写）
    const double fx = 312.99588161843817;     // X轴焦距
    const double fy = 312.9214941782706;     // Y轴焦距
    const double cx = 638.7854720110728;     // 主点X坐标
    const double cy = 514.5354880999439;     // 主点Y坐标
    const double xi = -0.17918626779269324;      // 球心偏移参数
    const double alpha = 0.590840604914911;   // 投影混合参数

    // 执行去畸变
    Mat undistorted_img = undistortDoubleSphere(distorted_img, 
                                               fx, fy, cx, cy, 
                                               xi, alpha);

    // 显示结果
    imshow("Distorted Image", distorted_img);
    imshow("Undistorted Image", undistorted_img);
    waitKey(0);

    // 保存结果
    // imwrite("undistorted_result.jpg", undistorted_img);
    imwrite("dscm7.png", undistorted_img);

    return 0;
}