
#include <iostream>
#include <cmath>

// 函数：根据FOV和传感器宽度计算焦距
double calculateFocalLength(double FOV, double sensorWidth) {
    // 将FOV转换为弧度
    double FOV_rad = FOV * M_PI / 180.0;
    
    // 计算焦距
    double focalLength = sensorWidth / (2 * tan(FOV_rad / 2));
    
    return focalLength;
}

double calculateFocalLengthRad(double FOV, double sensorWidth) {
    // 将FOV转换为弧度
    double FOV_rad = FOV;
    
    // 计算焦距
    double focalLength = sensorWidth / (2 * tan(FOV_rad / 2));
    
    return focalLength;
}

int main() {
    // 输入：视场角（度），传感器宽度（毫米）
    double FOV =2.9670597, sensorWidth = 640.0;

    std::cout << "tan(FOV/2) = " << tan(FOV / 2) << std::endl;
    
    // 计算焦距
    double focalLength = calculateFocalLengthRad(FOV, sensorWidth);
    
    std::cout << "计算得到的焦距为： " << focalLength << std::endl;
    
    return 0;
}
