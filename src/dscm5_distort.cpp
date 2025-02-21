
//@date 2025-2-19
//@auther: wxliu
//@brief: DSCM: Double Sphere Camera Model

// distort image: 1. pinhole unprojection 2. ds projection
// use different fxs.

// deepseek

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

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

int main(int argc, char **argv) {
  #if 0  
    // 示例：将3D点投影到图像平面
    double x = 1.0, y = 2.0, z = 10.0; // 3D点坐标
    double fx = 500.0, fy = 500.0;     // 焦距
    double cx = 320.0, cy = 240.0;     // 主点坐标
    double xi = 0.5, alpha = 0.6;      // 双球模型参数

    // 调用投影函数
    vector<double> pixel_coords = doubleSphereProjection(x, y, z, fx, fy, cx, cy, xi, alpha);

    if (pixel_coords[0] != -1 && pixel_coords[1] != -1) {
        cout << "Projected pixel coordinates: (" << pixel_coords[0] << ", " << pixel_coords[1] << ")" << endl;
    } else {
        cout << "Invalid projection (point is outside the valid projection region)." << endl;
    }
  #endif
    
    Mat src = imread("/home/lwx/dev/cpp_ws/dscm/assets/image_L2.png"); // 读取无畸变图像
    if (src.empty()) return -1;

    // 双球模型参数（示例值，需根据实际标定结果填写）
    // double fx = 27.9964, fy = 27.9964, cx = 320.0, cy = 240.0, xi = -0.2776869842961572, alpha = 0.5635725608138804;
    double fx = 160.9964, fy = 160.9964, cx = 320.0, cy = 240.0, xi = -0.2776869842961572, alpha = 0.5635725608138804;
    double pinholeFx = 27.9964;  // pinhole 焦距 x
    double pinholeFy = 27.9964;  // pinhole 焦距 y

    Mat dst(src.size(), src.type(), Scalar(0));
    for (int v_src = 0; v_src < src.rows; ++v_src) 
    {
        for (int u_src = 0; u_src < src.cols; ++u_src) 
        {
            // Step 1: 计算归一化坐标
            // double mx = (u_src - cx) / fx;
            // double my = (v_src - cy) / fy;
            double mx = (u_src - cx) / pinholeFx;
            double my = (v_src - cy) / pinholeFy;
            double mz = 1.0;
            vector<double> pixel_coords = doubleSphereProjection(mx, my, mz, fx, fy, cx, cy, xi, alpha);

            if (pixel_coords[0] >= 0 && pixel_coords[1] >= 0 && pixel_coords[0] < dst.cols && pixel_coords[1] < dst.rows) {
                dst.at<Vec3b>((int) pixel_coords[1], (int) pixel_coords[0]) = src.at<Vec3b>(v_src, u_src);
            } else {
                // dst.at<Vec3b>((int) pixel_coords[1], (int) pixel_coords[0]) = 0;
            }
        }
    }

    imwrite("dscm5.png", dst);

    {
        int u_src = 0;
        int v_src = 0;
        for(; v_src < 4; v_src++)
        {
            for(u_src = 0; u_src < 4; u_src++)
            {
                std::cout << "u_src: " << u_src << ", v_src: " << v_src << std::endl;
                double x = (u_src - cx) / pinholeFx;
                double y = (v_src - cy) / pinholeFy;
                double z = 1.0;
                std::cout << "x: " << x << ", y: " << y << ", z: " << z << std::endl;

                // 公式 (41): 计算 d1
                double d1 = sqrt(x * x + y * y + z * z);

                // 公式 (42): 计算 d2
                double d2 = sqrt(x * x + y * y + (xi * d1 + z) * (xi * d1 + z));

                // 公式 (43): 检查投影有效性
                double w1 = (alpha <= 0.5) ? (alpha / (1 - alpha)) : ((1 - alpha) / alpha);
                double w2 = (w1 + xi) / sqrt(2 * w1 * xi + xi * xi + 1);

                if (z <= -w2 * d1) {
                    // 无效投影，返回 (-1, -1)
                    // std::cerr << "Point out of valid projection range!" << std::endl;
                    std::cout << "Point out of valid projection range!" << std::endl;
                    // return -1;
                    continue;
                }

                // 公式 (40): 计算投影坐标
                double denominator = alpha * d2 + (1 - alpha) * (xi * d1 + z);
                if (fabs(denominator) < 1e-6) {
                    // 避免除以零
                    std::cout << "denominator is zero" << std::endl;
                    // return -1;
                    continue;
                }

                double fx_new = pinholeFx * (x / z) / (x / denominator);
                std::cout << "fx_new: " << fx_new << std::endl;

                // double u = fx * (x / denominator) + cx;
                // double v = fy * (y / denominator) + cy;
            }
        }
        
    }

    return 0;
}