
//@date 2025-2-19
//@auther: wxliu
//@brief: DSCM: Double Sphere Camera Model

// distort image: 1. ds unprojection 2. pinhole projection
// use the same fx.

// deepseek

#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;

Mat applyDoubleSphereDistortion(const Mat& src, double fx, double fy, double cx, double cy, double xi, double alpha) {
    Mat dst(src.size(), src.type(), Scalar(0));
    Mat map_x(src.size(), CV_32F);
    Mat map_y(src.size(), CV_32F);

    // 预计算映射关系
    for (int v = 0; v < src.rows; ++v) {
        for (int u = 0; u < src.cols; ++u) {
            // 步骤1：双球模型逆投影到3D射线
            double mx = (u - cx) / fx;
            double my = (v - cy) / fy;
            double r2 = mx*mx + my*my;
            
            // 检查有效投影区域（公式51）
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

            double x = scale * mx;
            double y = scale * my;
            double z = scale * mz - xi;

            // 归一化向量
            double norm = sqrt(x*x + y*y + z*z);
            if (norm < 1e-6) {
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }
            x /= norm;
            y /= norm;
            z /= norm;

            // 步骤2：投影到原针孔模型
            if (z <= 1e-6) { // 避免除以零
                map_x.at<float>(v, u) = -1;
                map_y.at<float>(v, u) = -1;
                continue;
            }
            
            double u_src = fx * (x / z) + cx;
            double v_src = fy * (y / z) + cy;

            map_x.at<float>(v, u) = u_src;
            map_y.at<float>(v, u) = v_src;
        }
    }

    // 使用remap进行高效重映射
    remap(src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

    return dst;
}

int main() {
    // 读取输入图像
    Mat src = imread("/home/lwx/dev/cpp_ws/dscm/assets/image_L2.png");
    if (src.empty()) {
        std::cerr << "Error: Could not load image" << std::endl;
        return -1;
    }

    // 双球模型参数（需根据实际标定结果调整）
    // double fx = 500.0, fy = 500.0;  // 焦距
    double fx = 27.9964, fy = 27.9964;  // 焦距
    double cx = src.cols/2.0, cy = src.rows/2.0; // 光心
    double xi = 0.5;    // 球心偏移参数
    double alpha = 0.6; // 混合参数

    // 应用畸变
    Mat distorted = applyDoubleSphereDistortion(src, fx, fy, cx, cy, xi, alpha);

    // 显示结果
    imshow("Original", src);
    imshow("Distorted", distorted);
    waitKey();

    return 0;
}