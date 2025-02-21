
//@date 2025-2-19
//@auther: wxliu
//@brief: DSCM: Double Sphere Camera Model

// distort image: 1. ds unprojection 2. pinhole projection
// use the same fx.


#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;

Mat transformImageToDoubleSphere(const Mat& src, double fx, double fy, double cx, double cy, double xi, double alpha) {
    Mat dst(src.size(), src.type(), Scalar(0));

    int err1 = 0, err2 = 0, err3 = 0, err4 = 0, err5 = 0, err6 = 0;
    for (int v_dst = 0; v_dst < dst.rows; ++v_dst) {
        for (int u_dst = 0; u_dst < dst.cols; ++u_dst) {
            // Step 1: 计算归一化坐标
            double mx = (u_dst - cx) / fx;
            double my = (v_dst - cy) / fy;
            double r2 = mx * mx + my * my;

            // 检查投影有效性（公式51）
            if (alpha > 0.5 && r2 > 1.0 / (2 * alpha - 1)) {
                err1++;
                continue;
            }

            // Step 2: 计算mz（公式50）
            double sqrt_term = sqrt(1 - (2 * alpha - 1) * r2);
            double denominator_alpha = alpha * sqrt_term + (1 - alpha);
            if (denominator_alpha <= 1e-6) {err2++;continue;}
            double mz = (1 - alpha * alpha * r2) / denominator_alpha;

            // Step 3: 计算3D方向向量（公式46）
            double numerator = mz * xi + sqrt(mz * mz + (1 - xi * xi) * r2);
            double denominator_total = mz * mz + r2;
            if (denominator_total <= 1e-6) {err3++;continue;}
            double scale = numerator / denominator_total;

            double x = scale * mx;
            double y = scale * my;
            double z = scale * mz - xi;

            // 归一化向量
            double norm = sqrt(x * x + y * y + z * z);
            if (norm <= 1e-6) {err4++;continue;}
            /* temp comment*/
            x /= norm;
            y /= norm;
            z /= norm;
            

            // Step 4: 投影到针孔模型（原图像）
            if (z <= 1e-6) {err5++;continue;} // 避免除以零
            double u_src = fx * (x / z) + cx;
            double v_src = fy * (y / z) + cy;

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
            else {err6++;}
        }
    }

    std::cout << "err1: " << err1 << ", err2: " << err2 << ", err3: " << err3 << ", err4: " << err4 << ", err5: " << err5 << ", err6: " << err6 << std::endl;

    return dst;
}


int main() {

#if 0
    // Mat src = imread("/home/lwx/dev/cpp_ws/dscm/assets/image_L2.png"); // 读取无畸变图像
    Mat src = imread("/home/lwx/dev/cpp_ws/dscm/assets/image_L3.png"); // 读取无畸变图像
#else    
    Mat src = imread("/home/lwx/dev/cpp_ws/dscm/assets/dscm6.png"); // 读取无畸变图像
#endif    
    if (src.empty()) return -1;

    // 双球模型参数（示例值，需根据实际标定结果填写）
#if 0    
    // double fx = 27.9964, fy = 27.9964, cx = 320.0, cy = 240.0, xi = -0.2776869842961572, alpha = 0.5635725608138804;
    double fx = 160, fy = 160, cx = 320.0, cy = 240.0, xi = -0.2776869842961572, alpha = 0.5635725608138804;
#else
    double fx = 312.99588161843817, fy = 312.9214941782706, cx = 638.7854720110728, cy = 514.5354880999439, xi = -0.17918626779269324, alpha = 0.590840604914911;
#endif
    Mat dst = transformImageToDoubleSphere(src, fx, fy, cx, cy, xi, alpha);
#if 0    
    imwrite("output_ds.png", dst);
#else    
    imwrite("output_ds2.png", dst);
#endif    

    // imshow("output_ds", dst);
    // waitKey(0);

    return 0;
}

// deepseek