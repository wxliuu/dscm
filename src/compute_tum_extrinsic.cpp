
#include <iostream>

#include <Eigen/Dense>

#include <iomanip>

/*
"T_imu_cam": [
            {
                "px": 0.04548094812071685,
                "py": -0.07145370002838907,
                "pz": -0.046315428444919249,
                "qx": -0.013392900690257393,
                "qy": -0.6945866755293793,
                "qz": 0.7192437840259219,
                "qw": 0.007639340823570553
            },
            {
                "px": -0.05546984222234079,
                "py": -0.06999334244486549,
                "pz": -0.049092582974927929,
                "qx": -0.01340980138125811,
                "qy": -0.7115668842588793,
                "qz": 0.7024477338114514,
                "qw": 0.007741299385907546
            }
        ]
*/

int main(int argc, char* argv[]) {
    
    Eigen::Quaterniond q_bc0(0.007639340823570553, -0.013392900690257393, -0.6945866755293793, 0.7192437840259219);
    Eigen::Quaterniond q_bc1(0.007741299385907546, -0.01340980138125811, -0.7115668842588793, 0.7024477338114514);
    q_bc0.normalize();
    q_bc1.normalize();

    Eigen::Matrix3d rotation_matrix0 = q_bc0.toRotationMatrix();  // 直接转换
    Eigen::Matrix3d rotation_matrix1 = q_bc1.toRotationMatrix();  // 直接转换

    Eigen::Vector3d p_bc0(0.04548094812071685, -0.07145370002838907, -0.046315428444919249);
    Eigen::Vector3d p_bc1(-0.05546984222234079, -0.06999334244486549, -0.049092582974927929);
    Eigen::Matrix4d T_bc0, T_bc1;
    T_bc0 << rotation_matrix0(0, 0), rotation_matrix0(0, 1), rotation_matrix0(0, 2), p_bc0(0),
             rotation_matrix0(1, 0), rotation_matrix0(1, 1), rotation_matrix0(1, 2), p_bc0(1),
             rotation_matrix0(2, 0), rotation_matrix0(2, 1), rotation_matrix0(2, 2), p_bc0(2),
             0.0, 0.0, 0.0, 1.0;
    T_bc1 << rotation_matrix1(0, 0), rotation_matrix1(0, 1), rotation_matrix1(0, 2), p_bc1(0),
             rotation_matrix1(1, 0), rotation_matrix1(1, 1), rotation_matrix1(1, 2), p_bc1(1),
             rotation_matrix1(2, 0), rotation_matrix1(2, 1), rotation_matrix1(2, 2), p_bc1(2),
             0.0, 0.0, 0.0, 1.0;

    // 计算 T_c1_c0 = T_bc1.inverse() * T_bc0
    Eigen::Matrix4d T_c1_c0 = T_bc1.inverse() * T_bc0;

    std::cout << std::fixed << std::setprecision(13) << "T_c1_c0=\n" << T_c1_c0 << std::endl;

    // std::cout << std::fixed << std::setprecision(13) << "rotation_matrix0=\n" << rotation_matrix0 << std::endl;
    // std::cout << std::fixed << std::setprecision(13) << "\nrotation_matrix1=\n" << rotation_matrix1 << std::endl;
    
    return 0;
}