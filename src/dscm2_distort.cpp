
//@date 2025-2-19
//@auther: wxliu
//@brief: DSCM: Double Sphere Camera Model

// distort image: 1. ds unprojection 2. pinhole projection
// use different fx.


#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

/**
 * 双球体模型参数结构体
 */
struct DoubleSphereParams {
    double fx;     // 焦距 x
    double fy;     // 焦距 y
    double cx;     // 主点 x
    double cy;     // 主点 y
    double xi;     // 双球体参数
    double alpha;  // 权重参数 (0 到 1 之间)
};

/**
 * 将像素点投影到 pinhole 模型
 *
 * @param point 3D 点
 * @param pinhole pinhole 模型参数
 * @return 2D 像素坐标
 */
Point2d ProjectToPinhole(const Point3d& point, double fx, double fy, double cx, double cy) {
    double x = point.x;
    double y = point.y;
    double z = point.z;

    double norm = sqrt(x * x + y * y + z * z);
    if (norm <= 1e-6) Point2d(-1, -1);

    if (z <= 1e-6) {
        return Point2d(-1, -1); // 避免除以零
    }

    double u = fx * x / z + cx;
    double v = fy * y / z + cy;

    return Point2d(u, v);
}

/**
 * 双球体模型逆投影
 *
 * @param u 图像坐标 x
 * @param v 图像坐标 y
 * @param ds 双球体模型参数
 * @return 3D 点
 */
Point3d InverseDoubleSphereProjection(double u, double v, const DoubleSphereParams& ds) {
    double mx = (u - ds.cx) / ds.fx;
    double my = (v - ds.cy) / ds.fy;
    double r_squared = mx * mx + my * my;

    // 检查投影有效性（公式51）
    if (ds.alpha > 0.5 && r_squared > 1.0 / (2 * ds.alpha - 1)) {
        return Point3d(0, 0, 0);
    }

    double numerator_mz = 1.0 - ds.alpha * ds.alpha * r_squared;
    double denominator_mz = ds.alpha * sqrt(1.0 - (2 * ds.alpha - 1) * r_squared) + (1.0 - ds.alpha);
    if (denominator_mz <= 1e-6) {
        return Point3d(0, 0, 0); // 避免除以零
    }
    double mz = numerator_mz / denominator_mz;

    double numerator_scale = ds.xi * mz + sqrt(mz * mz + (1.0 - ds.xi * ds.xi) * r_squared);
    double denominator_scale = mz * mz + r_squared;
    if (denominator_scale <= 1e-6) {
        return Point3d(0, 0, 0); // 避免除以零
    }
    double scale = numerator_scale / denominator_scale;

    double x = scale * mx;
    double y = scale * my;
    double z = scale * mz - ds.xi;

    return Point3d(x, y, z);
}

/**
 * 模拟双球体模型畸变
 *
 * @param inputImage 输入的无畸变图像
 * @param ds 双球体模型参数
 * @param pinhole pinhole 模型参数
 * @return 模拟畸变后的图像
 */
Mat SimulateDistortion(const Mat& inputImage, const DoubleSphereParams& ds, double pinholeFx, double pinholeFy, double pinholeCx, double pinholeCy) {
    Mat outputImage(inputImage.rows, inputImage.cols, inputImage.type());
    Mat map_x(inputImage.rows, inputImage.cols, CV_32FC1);
    Mat map_y(inputImage.rows, inputImage.cols, CV_32FC1);

    // 创建映射
    for (int v = 0; v < inputImage.rows; ++v) {
        for (int u = 0; u < inputImage.cols; ++u) {
            // 双球体模型下像素到 3D 点的逆投影
            Point3d worldPoint = InverseDoubleSphereProjection(u, v, ds);

            // 3D 点投影到 pinhole 模型
            Point2d pinholePoint = ProjectToPinhole(worldPoint, pinholeFx, pinholeFy, pinholeCx, pinholeCy);

            // 记录映射
            map_x.at<float>(v, u) = pinholePoint.x;
            map_y.at<float>(v, u) = pinholePoint.y;
        }
    }

    // 应用映射
    remap(inputImage, outputImage, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT);

    return outputImage;
}

Mat SimulateDistortion2(const Mat& inputImage, const DoubleSphereParams& ds, double pinholeFx, double pinholeFy, double pinholeCx, double pinholeCy) {
    // Mat outputImage(inputImage.rows, inputImage.cols, inputImage.type(), Scalar(0));
    Mat outputImage(inputImage.size(), inputImage.type(), Scalar(0));
    // imshow("outputImage", outputImage);
    // waitKey(0);

    // 创建映射
    for (int v = 0; v < inputImage.rows; ++v) {
        for (int u = 0; u < inputImage.cols; ++u) {
            // 双球体模型下像素到 3D 点的逆投影
            Point3d worldPoint = InverseDoubleSphereProjection(u, v, ds);

            // 3D 点投影到 pinhole 模型
            Point2d pinholePoint = ProjectToPinhole(worldPoint, pinholeFx, pinholeFy, pinholeCx, pinholeCy);

            // 双线性插值
            if (pinholePoint.x >= 0 && pinholePoint.x < inputImage.cols - 1 && pinholePoint.y >= 0 && pinholePoint.y < inputImage.rows - 1) {
                int x0 = static_cast<int>(pinholePoint.x);
                int y0 = static_cast<int>(pinholePoint.y);
                float dx = pinholePoint.x - x0;
                float dy = pinholePoint.y - y0;

                Vec3b p00 = inputImage.at<Vec3b>(y0, x0);
                Vec3b p01 = inputImage.at<Vec3b>(y0, x0 + 1);
                Vec3b p10 = inputImage.at<Vec3b>(y0 + 1, x0);
                Vec3b p11 = inputImage.at<Vec3b>(y0 + 1, x0 + 1);

                Vec3b interpolated = 
                    p00 * (1 - dx) * (1 - dy) +
                    p01 * dx * (1 - dy) +
                    p10 * (1 - dx) * dy +
                    p11 * dx * dy;

                outputImage.at<Vec3b>(v, u) = interpolated;
            }
            // else{
            //     std::cout << "pinholePoint.x: " << pinholePoint.x << ", pinholePoint.y: " << pinholePoint.y << std::endl;
            //     outputImage.at<Vec3b>(v, u) = Vec3b(0, 0, 0);
            // }
        }
    }


    return outputImage;
}

int main() {
    // 示例参数（根据实际情况更换）
    // DoubleSphereParams ds = {27.9964, 27.9964, 320, 240, 0.8, 0.6}; // fx, fy, cx, cy, xi, alpha
    DoubleSphereParams ds = {160.9964, 160.9964, 320, 240, -0.2776869842961572, 0.5635725608138804}; // fx, fy, cx, cy, xi, alpha
    // DoubleSphereParams ds = {192.9964, 192.9964, 320, 240, -0.2776869842961572, 0.5635725608138804}; // fx, fy, cx, cy, xi, alpha
    double pinholeFx = 27.9964;  // pinhole 焦距 x
    double pinholeFy = 27.9964;  // pinhole 焦距 y
    double pinholeCx = 320;  // pinhole 主点 x
    double pinholeCy = 240;  // pinhole 主点 y

    // 加载无畸变图像
    // Mat inputImage = imread("undistorted_image.jpg", IMREAD_COLOR);
    // Mat inputImage = imread("/home/lwx/dev/cpp_ws/dscm/assets/image_L2.png");
    Mat inputImage = imread("/home/lwx/dev/cpp_ws/dscm/assets/image_L2.png", IMREAD_COLOR);
    if (inputImage.empty()) {
        cerr << "Error: could not load image!" << endl;
        return -1;
    }

    // 模拟畸变
    Mat distortedImage = SimulateDistortion(inputImage, ds, pinholeFx, pinholeFy, pinholeCx, pinholeCy);
    // Mat distortedImage = SimulateDistortion2(inputImage, ds, pinholeFx, pinholeFy, pinholeCx, pinholeCy);

    // 显示和保存结果
    imshow("Original", inputImage);
    imshow("Distorted", distortedImage);
    imwrite("distorted_image.jpg", distortedImage);

    waitKey(0);
    return 0;
}

// kimi