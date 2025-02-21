
//@date 2025-2-19
//@auther: wxliu
//@brief: DSCM: Double Sphere Camera Model

// undistort image: 1. pinhole unprojection 2. ds projection
// use the same fx.

// wxliu

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

// #define _Nearest_Neighbor_Interpolation

// 双球模型投影函数
vector<double> doubleSphereProjection(double x, double y, double z, double fx, double fy, double cx, double cy, double xi, double alpha) {
    vector<double> pixel_coords(2, -1); // 初始化返回的像素坐标

    // 公式 (41): 计算 d1
    double d1 = sqrt(x * x + y * y + z * z);

    // 公式 (42): 计算 d2
    double d2 = sqrt(x * x + y * y + (xi * d1 + z) * (xi * d1 + z));

    // 公式 (43): 检查投影有效性
    double w1 = (alpha <= 0.5) ? (alpha / (1 - alpha)) : ((1 - alpha) / alpha);
    double w2 = (w1 + xi) / sqrt(2 * w1 * xi + xi * xi + 1);

    if (z <= -w2 * d1) {
        // 无效投影，返回 (-1, -1)
        return pixel_coords;
    }

    // 公式 (40): 计算投影坐标
    double denominator = alpha * d2 + (1 - alpha) * (xi * d1 + z);
    if (fabs(denominator) < 1e-6) {
        // 避免除以零
        return pixel_coords;
    }

    double u = fx * (x / denominator) + cx;
    double v = fy * (y / denominator) + cy;

    pixel_coords[0] = u;
    pixel_coords[1] = v;

    return pixel_coords;
}

int main(int argc, char **argv) 
{
#if 0    
    Mat src = imread("/home/lwx/dev/cpp_ws/dscm/assets/screen_20_double-sphere.jpg"); // 读取无畸变图像
#else    
    Mat src = imread("/home/lwx/dev/cpp_ws/dscm/assets/image_L3_distort.png"); // 读取无畸变图像
#endif

    if (src.empty()) return -1;

#if 0
    // 双球模型参数（示例值，需根据实际标定结果填写）
    // double fx = 27.9964, fy = 27.9964, cx = 320.0, cy = 240.0, xi = -0.2776869842961572, alpha = 0.5635725608138804;
    double fx = 312.99588161843817, fy = 312.9214941782706, cx = 638.7854720110728, cy = 514.5354880999439, xi = -0.17918626779269324, alpha = 0.590840604914911;
    // double pinholeFx = fx; // 27.9964;  // pinhole 焦距 x
    // double pinholeFy = fy; // 27.9964;  // pinhole 焦距 y

    double pinholeFx = 380.2064;  // pinhole 焦距 x
    double pinholeFy = 380.1064;  // pinhole 焦距 y

#else
    double fx = 160.9964, fy = 160.9964, cx = 320.0, cy = 240.0, xi = -0.2776869842961572, alpha = 0.5635725608138804;
    double pinholeFx = 160.9964;  // pinhole 焦距 x
    double pinholeFy = 160.9964;  // pinhole 焦距 y
#endif    

    Mat dst(src.size(), src.type(), Scalar(0));
    for (int v_dst = 0; v_dst < dst.rows; ++v_dst) 
    {
        for (int u_dst = 0; u_dst < dst.cols; ++u_dst)
        {
            // Step 1: 计算归一化坐标
            // double mx = (u_src - cx) / fx;
            // double my = (v_src - cy) / fy;
            double mx = (u_dst - cx) / pinholeFx;
            double my = (v_dst - cy) / pinholeFy;
            double mz = 1.0;
            vector<double> pixel_coords = doubleSphereProjection(mx, my, mz, fx, fy, cx, cy, xi, alpha);
        #if defined(_Nearest_Neighbor_Interpolation)
            if (pixel_coords[0] >= 0 && pixel_coords[1] >= 0 && pixel_coords[0] < dst.cols && pixel_coords[1] < dst.rows) {
                dst.at<Vec3b>(v_dst, u_dst) = src.at<Vec3b>((int) pixel_coords[1], (int) pixel_coords[0]);
            } else {
                dst.at<Vec3b>(v_dst, u_dst) = 0;
            }

        #else

            double u_src = pixel_coords[0];
            double v_src = pixel_coords[1];

            // 双线性插值
            if (u_src >= 0 && u_src < src.cols - 1 && v_src >= 0 && v_src < src.rows - 1) {
                int x0 = static_cast<int>(u_src);
                int y0 = static_cast<int>(v_src);
                float dx = u_src - x0;
                float dy = v_src - y0;

                Vec3b p00 = src.at<Vec3b>(y0, x0);
                Vec3b p01 = src.at<Vec3b>(y0, x0 + 1);
                Vec3b p10 = src.at<Vec3b>(y0 + 1, x0);
                Vec3b p11 = src.at<Vec3b>(y0 + 1, x0 + 1);

                Vec3b interpolated = 
                    p00 * (1 - dx) * (1 - dy) +
                    p01 * dx * (1 - dy) +
                    p10 * (1 - dx) * dy +
                    p11 * dx * dy;

                dst.at<Vec3b>(v_dst, u_dst) = interpolated;
            }
        #endif    
        }
    }

#if defined(_Nearest_Neighbor_Interpolation)
    imwrite("dscm66.png", dst);
#else    
    imwrite("dscm67.png", dst);
#endif

    return 0;
}