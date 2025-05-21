// Do not undistort the images; perform stereo rectification directly instead.
// Skip undistortion and directly apply stereo rectification.
// Omit the undistortion step and proceed directly to stereo rectification.
// Skip undistortion and directly perform stereo rectification.

#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
using namespace cv;

#define _USE_THE_SAME_INTRINSICS_

#define _EUROC_DATASET_
// #define _TUM_DATASET_

#if defined(_EUROC_DATASET_) && !defined(_TUM_DATASET_)
int image_width = 752, image_height = 480;

// ds cam0
double fx0 = 349.7560023050409, fy0 = 348.72454229977037, cx0 = 365.89440762590149, cy0 = 249.32995565708704, xi0 = -0.2409573942178872, alpha0 = 0.566996899163044;

// ds cam1
double fx1 = 361.6713883800533, fy1 = 360.5856493689301, cx1 = 379.40818394080869, cy1 = 255.9772968522045, xi1 = -0.21300835384809328, alpha1 = 0.5767008625037023;

// R_cam1_cam0
cv::Mat extrinsic_R = ( cv::Mat_<double> ( 3,3 ) <<  0.9999972564779, 0.002312067192424, 0.0003760081024156,
-0.002317135723281, 0.9998980485066, 0.01408983584665,
-0.0003433931205242, -0.01409066845271, 0.9999006626377);


cv::Mat extrinsic_t = ( cv::Mat_<double> ( 3,1 ) << -0.1100738081272,
0.0003991215470141,
-0.000853702503358 );

#else // TUM dataset

int image_width = 512, image_height = 512;

// ds cam0
double fx0 = 158.28600034966977, fy0 = 158.2743455478755, cx0 = 254.96116578191653, cy0 = 256.8894394501779, xi0 = -0.17213086034353243, alpha0 = 0.5931177593944744;

// ds cam1
double fx1 = 157.91830144176309, fy1 = 157.8901286125632, cx1 = 252.56547609702953, cy1 = 255.02489416194656, xi1 = -0.17114780716007858, alpha1 = 0.5925543396658507;

// R_cam1_cam0
cv::Mat extrinsic_R = ( cv::Mat_<double> ( 3,3 ) << 0.9999995881755, -0.0008429104329, -0.0003363788793,
                                                    0.0008258897264, 0.9988593352731, -0.0477424993476,
                                                    0.0003762378345, 0.0477422018742, 0.9988596200695);


cv::Mat extrinsic_t = ( cv::Mat_<double> ( 3,1 ) << -0.1009678603465,
                                                    -0.0019668632282,
                                                    -0.0015905749062 );

#endif


double k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;
const cv::Mat D_left = ( cv::Mat_<double> ( 5,1 ) << k1, k2, p1, p2, 0.000000);
const cv::Mat D_right = ( cv::Mat_<double> ( 5,1 ) << k1, k2, p1, p2, 0.000000);

cv::Mat map_x_cam0_final, map_y_cam0_final, map_x_cam1_final, map_y_cam1_final;


void stereoRectifyAndUndistort() {
    // 1. Read the camera calibration parameters
    // 2. Read the stereo calibration parameters
    // 3. Rectify the stereo images
    // 4. Undistort the stereo images
    // 5. Display the stereo images


    cv::Mat R1, R2, P1, P2, Q;
    cv::Mat map_stereorectify_cam0_x, map_stereorectify_cam0_y, map_stereorectify_cam1_x, map_stereorectify_cam1_y;

#if defined(_USE_THE_SAME_INTRINSICS_)
    Mat cameraMatrixL = (Mat_<double>(3,3) << fx0, 0, cx0, 0, fy0, cy0, 0, 0, 1);
    Mat cameraMatrixR = (Mat_<double>(3,3) << fx1, 0, cx1, 0, fy1, cy1, 0, 0, 1);
#else // use respective intrinsic parameters
    Mat cameraMatrixL = K_left;
    Mat cameraMatrixR = K_right;
#endif

    cv::stereoRectify(cameraMatrixL, D_left, cameraMatrixR, D_right, cv::Size(image_width, image_height), extrinsic_R, extrinsic_t, R1, R2, P1, P2, Q);
    cv::initUndistortRectifyMap(cameraMatrixL, D_left, R1, P1, cv::Size(image_width, image_height), CV_32FC1, map_stereorectify_cam0_x, map_stereorectify_cam0_y);
    cv::initUndistortRectifyMap(cameraMatrixR, D_right, R2, P2, cv::Size(image_width, image_height), CV_32FC1, map_stereorectify_cam1_x, map_stereorectify_cam1_y);

    // img0
    // Mat map_x(img0.size(), CV_32F);
    // Mat map_y(img0.size(), CV_32F);

    Mat map_undistort_cam0_x(cv::Size(image_width, image_height), CV_32FC1);
    Mat map_undistort_cam0_y(cv::Size(image_width, image_height), CV_32FC1);

    Mat map_undistort_cam1_x(image_height, image_width, CV_32FC1);
    Mat map_undistort_cam1_y(image_height, image_width, CV_32FC1);

    for (int v = 0; v < image_height; ++v) {
        for (int u = 0; u < image_width; ++u) {
            // ------------------------------
            // Step 1: 计算无畸变坐标系的3D射线（针孔模型）
            // ------------------------------
            double x = (u - cameraMatrixL.at<double>(0, 2)) / cameraMatrixL.at<double>(0, 0);  // 归一化x坐标
            double y = (v - cameraMatrixL.at<double>(1, 2)) / cameraMatrixL.at<double>(1, 1);  // 归一化y坐标
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
                map_undistort_cam0_x.at<float>(v, u) = -1;
                map_undistort_cam0_y.at<float>(v, u) = -1;
                continue;
            }

            // 公式42: 计算d2
            double term_z = xi0*d1 + z;
            double d2 = sqrt(x*x + y*y + term_z*term_z);

            // 公式40: 计算畸变坐标
            double denominator = alpha0*d2 + (1-alpha0)*term_z;
            if (fabs(denominator) < 1e-9) {
                map_undistort_cam0_x.at<float>(v, u) = -1;
                map_undistort_cam0_y.at<float>(v, u) = -1;
                continue;
            }

            double u_dist = fx0*(x/denominator) + cx0;
            double v_dist = fy0*(y/denominator) + cy0;

            // ------------------------------
            // Step 3: 建立映射关系
            // ------------------------------
            // 注意这里映射方向是 undistorted -> distorted
            map_undistort_cam0_x.at<float>(v, u) = static_cast<float>(u_dist);
            map_undistort_cam0_y.at<float>(v, u) = static_cast<float>(v_dist);

        }
    }

    // Mat left_rectified, right_rectified;
    // cv::remap(img0, left_rectified, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));


    // img1
    for (int v = 0; v < image_height; ++v) {
        for (int u = 0; u < image_width; ++u) {
            // ------------------------------
            // Step 1: 计算无畸变坐标系的3D射线（针孔模型）
            // ------------------------------
            double x = (u - cameraMatrixR.at<double>(0, 2)) / cameraMatrixR.at<double>(0, 0);  // 归一化x坐标
            double y = (v - cameraMatrixR.at<double>(1, 2)) / cameraMatrixR.at<double>(1, 1);  // 归一化y坐标
            double z = 1.0;            // 假设深度为1

            // ------------------------------
            // Step 2: 应用双球模型正向投影（公式40-45）
            // ------------------------------
            // 公式41: 计算d1
            double d1 = sqrt(x*x + y*y + z*z);

            // 公式43-45: 检查有效投影区域
            double w1 = (alpha1 <= 0.5) ? (alpha1/(1-alpha1)) : ((1-alpha1)/alpha1);
            double w2 = (w1 + xi1)/sqrt(2*w1*xi1 + xi1*xi1 + 1);
            if (z <= -w2*d1) {
                map_undistort_cam1_x.at<float>(v, u) = -1;
                map_undistort_cam1_y.at<float>(v, u) = -1;
                continue;
            }

            // 公式42: 计算d2
            double term_z = xi1*d1 + z;
            double d2 = sqrt(x*x + y*y + term_z*term_z);

            // 公式40: 计算畸变坐标
            double denominator = alpha1*d2 + (1-alpha1)*term_z;
            if (fabs(denominator) < 1e-9) {
                map_undistort_cam1_x.at<float>(v, u) = -1;
                map_undistort_cam1_y.at<float>(v, u) = -1;
                continue;
            }

            double u_dist = fx1*(x/denominator) + cx1;
            double v_dist = fy1*(y/denominator) + cy1;

            // ------------------------------
            // Step 3: 建立映射关系
            // ------------------------------
            // 注意这里映射方向是 undistorted -> distorted
            map_undistort_cam1_x.at<float>(v, u) = static_cast<float>(u_dist);
            map_undistort_cam1_y.at<float>(v, u) = static_cast<float>(v_dist);

        }
    }

    // cv::remap(img1, right_rectified, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::remap(map_undistort_cam0_x, map_x_cam0_final, map_stereorectify_cam0_x, map_stereorectify_cam0_y, cv::INTER_CUBIC);
    cv::remap(map_undistort_cam0_y, map_y_cam0_final, map_stereorectify_cam0_x, map_stereorectify_cam0_y, cv::INTER_CUBIC);
    cv::remap(map_undistort_cam1_x, map_x_cam1_final, map_stereorectify_cam1_x, map_stereorectify_cam1_y, cv::INTER_CUBIC);
    cv::remap(map_undistort_cam1_y, map_y_cam1_final, map_stereorectify_cam1_x, map_stereorectify_cam1_y, cv::INTER_CUBIC);

    // Mat canvas;
    // hconcat(left_rectified, right_rectified, canvas);
    // for (int i = 0; i < canvas.rows; i += 32)
    //     line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1);

    // imshow("Rectified", canvas);
    // waitKey(0);

}

struct DoubleSphereParams {
    double fx, fy, cx, cy, xi, alpha;
};

cv::Point3d unproject(double u, double v, const DoubleSphereParams& params) {
    double mx = (u - params.cx) / params.fx;
    double my = (v - params.cy) / params.fy;
    double r_sq = mx*mx + my*my;

    if (params.alpha > 0.5) {
        double threshold = 1.0 / (2 * params.alpha - 1);
        if (r_sq > threshold) {
            std::cout << "unproject Invalid point: (" << u << ", " << v << ") exceeds threshold." << std::endl;
            return cv::Point3d(0, 0, 0);
        }
    }

    double numerator = 1.0 - params.alpha * params.alpha * r_sq;
    double denominator = params.alpha * sqrt(1.0 - (2*params.alpha - 1)*r_sq) + (1 - params.alpha);
    double mz = numerator / denominator;

    double xi = params.xi;
    double sqrt_term = sqrt(mz*mz + (1 - xi*xi)*r_sq);
    double coefficient = (mz * xi + sqrt_term) / (mz*mz + r_sq);

    double x = mx * coefficient;
    double y = my * coefficient;
    double z = mz * coefficient - xi;

    double norm = sqrt(x*x + y*y + z*z);
    if (fabs(norm) < 1e-9) {
        std::cout << "unproject norm=" << norm << std::endl;
        return cv::Point3d(0, 0, 0);
    }
    return cv::Point3d(x/norm, y/norm, z/norm);
}

cv::Point2d project(const cv::Point3d& point, const DoubleSphereParams& params) {
    double x = point.x;
    double y = point.y;
    double z = point.z;

    double d1 = sqrt(x*x + y*y + z*z);
    double xi = params.xi;

    double d2 = sqrt(x*x + y*y + (xi*d1 + z)*(xi*d1 + z));

    double denominator = params.alpha*d2 + (1 - params.alpha)*(xi*d1 + z);
    if (fabs(denominator) <= 1e-6) {
        std::cout << "project Invalid point: (" << x << ", " << y << ", " << z << ") results in zero denominator." << std::endl;
        return cv::Point2d(-1, -1);
    }

    double u = params.fx * x / denominator + params.cx;
    double v = params.fy * y / denominator + params.cy;

    return cv::Point2d(u, v);
}

void computeRemap(const DoubleSphereParams& params, const cv::Mat& R, cv::Mat& mapX, cv::Mat& mapY) {
    int width = mapX.cols;
    int height = mapX.rows;

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            cv::Point3d dir_corrected = unproject(u, v, params);
            if (dir_corrected == cv::Point3d(0, 0, 0)) {
                mapX.at<float>(v, u) = -1;
                mapY.at<float>(v, u) = -1;
                continue;
            }

            cv::Mat dir_mat = (cv::Mat_<double>(3,1) << dir_corrected.x, dir_corrected.y, dir_corrected.z);
            cv::Mat rotated_dir = R.t() * dir_mat;

            cv::Point3d rotated_point(rotated_dir.at<double>(0), rotated_dir.at<double>(1), rotated_dir.at<double>(2));
            cv::Point2d uv = project(rotated_point, params);

            mapX.at<float>(v, u) = uv.x;
            mapY.at<float>(v, u) = uv.y;
        }
    }
}

void computeRemap2(const DoubleSphereParams& params1, const DoubleSphereParams& params2, const cv::Mat& R, cv::Mat& mapX, cv::Mat& mapY) 
{
    int width = mapX.cols;
    int height = mapX.rows;

    for (int v = 0; v < height; ++v) 
    {
        for (int u = 0; u < width; ++u) 
        {
            cv::Point3d dir_corrected = unproject(u, v, params1);
            if (dir_corrected == cv::Point3d(0, 0, 0)) {
                // mapX.at<float>(v, u) = -1;
                // mapY.at<float>(v, u) = -1;
                std::cout << "Invalid point at (" << u << ", " << v << ")" << std::endl;
                continue;
            }

            cv::Mat dir_mat = (cv::Mat_<double>(3,1) << dir_corrected.x, dir_corrected.y, dir_corrected.z);
            cv::Mat rotated_dir = R * dir_mat;

            cv::Point3d rotated_point(rotated_dir.at<double>(0), rotated_dir.at<double>(1), rotated_dir.at<double>(2));
            cv::Point2d uv = project(rotated_point, params2);

            if (uv.x < 0 || uv.x >= width || uv.y < 0 || uv.y >= height) {
                // std::cout << "Projected point out of bounds: (" << uv.x << ", " << uv.y << ")" << std::endl;
                continue;
            }

            mapX.at<float>(uv.y, uv.x) = u;
            mapY.at<float>(uv.y, uv.x) = v;
        }
    }
}

void computeRemap3(const DoubleSphereParams& params1, const DoubleSphereParams& params2, const cv::Mat& R, cv::Mat& mapX, cv::Mat& mapY) 
{
    int width = mapX.cols;
    int height = mapX.rows;

    for (int dst_v = 0; dst_v < height; ++dst_v) 
    {
        for (int dst_u = 0; dst_u < width; ++dst_u) 
        {
            cv::Point3d dir_corrected = unproject(dst_u, dst_v, params2);
            if (dir_corrected == cv::Point3d(0, 0, 0)) {
                // mapX.at<float>(v, u) = -1;
                // mapY.at<float>(v, u) = -1;
                std::cout << "Invalid point at (" << dst_u << ", " << dst_v << ")" << std::endl;
                continue;
            }

            cv::Mat dir_mat = (cv::Mat_<double>(3,1) << dir_corrected.x, dir_corrected.y, dir_corrected.z);
            cv::Mat rotated_dir = R.t() * dir_mat;

            cv::Point3d rotated_point(rotated_dir.at<double>(0), rotated_dir.at<double>(1), rotated_dir.at<double>(2));
            cv::Point2d uv = project(rotated_point, params1);

            if (uv.x < 0 || uv.x >= width || uv.y < 0 || uv.y >= height) {
                // std::cout << "Projected point out of bounds: (" << uv.x << ", " << uv.y << ")" << std::endl;
                continue;
            }

            mapX.at<float>(dst_v, dst_u) = uv.x;
            mapY.at<float>(dst_v, dst_u) = uv.y;
        }
    }
}

int main() 
{
    if(0)
    {
        std::string image0Path = "/root/lwx_dataset/tum_test/1520530424604726258_cam0.png";
        std::string image1Path = "/root/lwx_dataset/tum_test/1520530424604726258_cam1.png";

        cv::Mat left_img = cv::imread(image0Path.c_str(), cv::IMREAD_GRAYSCALE);
        cv::Mat right_img = cv::imread(image1Path.c_str(), cv::IMREAD_GRAYSCALE);
        if (left_img.empty() || right_img.empty())
        {
            std::cerr << "image is empty! \npath=" << image0Path << std::endl;
            return -1;
        }

        return 0;
    }

    stereoRectifyAndUndistort();

    // 假设已从标定文件读取参数
    DoubleSphereParams left_params{ fx0, fy0, cx0, cy0, xi0, alpha0 };
    DoubleSphereParams right_params{ fx1, fy1, cx1, cy1, xi1, alpha1 };
    cv::Mat R = extrinsic_R, T = extrinsic_t; // 左右相机之间的外参

#if 0
    cv::Mat left_K = (cv::Mat_<double>(3,3) << left_params.fx, 0, left_params.cx,
                                               0, left_params.fy, left_params.cy,
                                               0, 0, 1);
    cv::Mat right_K = (cv::Mat_<double>(3,3) << right_params.fx, 0, right_params.cx,
                                                0, right_params.fy, right_params.cy,
                                                0, 0, 1);
    
    cv::Mat D_left, D_right; // 假设无畸变参数
    cv::Size image_size( image_width, image_height );

    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(left_K, D_left, right_K, D_right, image_size, R, T,
                      R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, image_size);

    // 创建映射表
    cv::Mat left_mapX, left_mapY, right_mapX, right_mapY;
    left_mapX.create(image_size, CV_32F);
    left_mapY.create(image_size, CV_32F);
    right_mapX.create(image_size, CV_32F);
    right_mapY.create(image_size, CV_32F);
    computeRemap(left_params, R1, left_mapX, left_mapY);
    computeRemap(right_params, R2, right_mapX, right_mapY);
#endif
    
    std::cout << "image_width: " << image_width << " image_height: " << image_height << std::endl;
    cv::Mat left_mapX(cv::Size(image_width, image_height), CV_32FC1);
    cv::Mat left_mapY(image_height, image_width, CV_32FC1);

#if 0
    computeRemap2(left_params, right_params, R, left_mapX, left_mapY);
#else
    computeRemap3(left_params, right_params, R, left_mapX, left_mapY);
#endif

#if defined(_EUROC_DATASET_)    
    // 读取图像并校正
    std::string filename = "1403715886584058112.png";
    std::string image0Path = "/root/lwx_dataset/EuRoC/V1_03_difficult/mav0/cam0/data/" + filename;
    std::string image1Path = "/root/lwx_dataset/EuRoC/V1_03_difficult/mav0/cam1/data/" + filename;
    cv::Mat left_img = cv::imread(image0Path.c_str(), cv::IMREAD_GRAYSCALE);
    cv::Mat right_img = cv::imread(image1Path.c_str(), cv::IMREAD_GRAYSCALE);
    // imshow("Left Image", left_img);
    // imshow("Right Image", right_img);
    // waitKey(0);
#else
    std::string image0Path = "/root/lwx_dataset/tum_test/1520530424604726258_cam0.png";
    std::string image1Path = "/root/lwx_dataset/tum_test/1520530424604726258_cam1.png";
    std::cout << "image0path=" << image0Path << std::endl;
    std::cout << "image1Path=" << image1Path << std::endl;

    cv::Mat left_img = cv::imread(image0Path, cv::IMREAD_GRAYSCALE);
    cv::Mat right_img = cv::imread(image1Path.c_str(), cv::IMREAD_GRAYSCALE);
    if (left_img.empty() || right_img.empty())
    {
        std::cerr << "image is empty! \npath=" << image0Path << std::endl;
        return -1;
    }
#endif

#if defined(_STEREO_RECTIFY_UNDISTORT_)    
    {
        Mat left_rectified, right_rectified;
        left_rectified = Mat::zeros(left_img.size(), left_img.type());
        right_rectified = Mat::zeros(right_img.size(), right_img.type()); // 这样对齐后，多余的部分应该就是黑色的区域

        cv::remap(left_img, left_rectified, map_x_cam0_final, map_y_cam0_final, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::remap(right_img, right_rectified, map_x_cam1_final, map_y_cam1_final, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        // cv::remap(img0, left_rectified, map_x_cam0_final, map_y_cam0_final, cv::INTER_LINEAR);
        // cv::remap(img1, right_rectified, map_x_cam1_final, map_y_cam1_final, cv::INTER_LINEAR);

        // 可视化结果
        Mat canvas;
        hconcat(left_rectified, right_rectified, canvas);
        for (int i = 0; i < canvas.rows; i += 32)
            cv::line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1);

        imshow("stereo_rectify_undistort", canvas);
        waitKey(0);
    }
#endif

    cv::Mat left_rectified, right_rectified;
    left_rectified = Mat::zeros(left_img.size(), left_img.type());
    cv::remap(left_img, left_rectified, left_mapX, left_mapY, cv::INTER_LINEAR);
    // cv::remap(right_img, right_rectified, right_mapX, right_mapY, cv::INTER_LINEAR);

    // 显示结果
    // cv::imshow("Left Rectified", left_rectified);
    // cv::imshow("Right Rectified", right_rectified);
    // cv::waitKey();

    // std::cout << "Left Rectified Image Size: " << left_rectified.size() << std::endl;
    // std::cout << "Right Rectified Image Size: " << right_rectified.size() << std::endl;
    // std::cout << "left_rectified.rows: " << left_rectified.rows << ", left_rectified.cols: " << left_rectified.cols << std::endl;
    // std::cout << "right_rectified.rows: " << right_rectified.rows << ", right_rectified.cols: " << right_rectified.cols << std::endl;
    // std::cout << "left_rectified.type: " << left_rectified.type() << std::endl;
    // std::cout << "right_rectified.type: " << right_rectified.type() << std::endl;
    // std::cout << "left_rectified.dimensions: " << left_rectified.dims << std::endl;
    // std::cout << "right_rectified.dimensions: " << right_rectified.dims << std::endl;

    Mat canvas;
    // hconcat(left_rectified, right_rectified, canvas);
    hconcat(left_rectified, right_img, canvas);
    for (int i = 0; i < canvas.rows; i += 32)
        cv::line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1);

    imshow("Rectified", canvas);

    if(1)
    {
        Mat canvas;
        hconcat(left_img, right_img, canvas);
        for (int i = 0; i < canvas.rows; i += 32)
            cv::line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1);

        imshow("raw", canvas);
    }

    waitKey(0);

    return 0;
}