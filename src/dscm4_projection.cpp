
//@date 2025-2-19
//@auther: wxliu
//@brief: DSCM: Double Sphere Camera Model

// chatgpt

#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

using namespace cv;

#include <cmath>
#include <iostream>

// 定义相机内参结构体
struct CameraParams {
    double fx, fy, cx, cy, xi, alpha;
};

// 定义3D点结构体
struct Point3D {
    double x, y, z;
};

// 定义2D点结构体
struct Point2D {
    double u, v;
};

// 计算3D点的距离
double distance3D(const Point3D& point) {
    return std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
}

// 计算有效范围的 w2
double computeW2(double alpha, double xi) {
    double w1 = (alpha <= 0.5) ? alpha / (1 - alpha) : (1 - alpha) / alpha;
    return (w1 + xi) / std::sqrt(2 * w1 * xi + xi * xi + 1);
}

// 双球模型投影函数
Point2D projectToImagePlane(const Point3D& point, const CameraParams& params) {
    // 计算第一个球体的距离 d1
    double d1 = distance3D(point);

    // 计算第二个球体的距离 d2
    double d2 = std::sqrt(point.x * point.x + point.y * point.y + std::pow(params.xi * d1 + point.z, 2));

    // 计算有效范围的 w2
    double w2 = computeW2(params.alpha, params.xi);

    // 检查是否满足投影的有效范围条件 (公式 43)
    if (point.z <= -w2 * d1) {
        std::cerr << "Point out of valid projection range!" << std::endl;
        return { -1, -1 };  // 返回一个无效的投影
    }

    // 计算分母部分
    double denominator = params.alpha * d2 + (1 - params.alpha) * (params.xi * d1 + point.z);

    // 计算投影后的2D点
    Point2D projectedPoint;
    projectedPoint.u = params.fx * (point.x / denominator) + params.cx;
    projectedPoint.v = params.fy * (point.y / denominator) + params.cy;

    return projectedPoint;
}

int main() {
    // 定义相机参数
    CameraParams cameraParams = { 500.0, 500.0, 320.0, 240.0, 0.5, 0.6 };
    
    // 定义一个3D点
    Point3D point3D = { 1.0, 2.0, 10.0 };

    // 计算投影
    Point2D projected = projectToImagePlane(point3D, cameraParams);
    
    // 如果投影有效，输出结果
    if (projected.u != -1 && projected.v != -1) {
        std::cout << "Projected 2D point: (" << projected.u << ", " << projected.v << ")\n";
    } else {
        std::cout << "Invalid point for projection.\n";
    }

    return 0;
}