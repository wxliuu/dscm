#pragma once

#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
// #include "Eigen/Core"
// #include <opencv2/gapi/core.hpp>
#include <iostream>
#include <iterator>

#include <stdio.h>
#include <stdlib.h>
///#include <arm_neon.h>
#include <math.h>

using namespace cv;
using namespace std;

// 2023-2-9
#include "wx_system.h"
using namespace wx;
// the end.

// wxliu added 2023-2-3
#ifdef __ARM__
#include <arm_neon.h>
#else
#include "src/util/tbb.h"
using namespace sv;
#define _WX_GRAIN_SIZE_ 1
#endif
// the end.

class undist_lite
{
// added by wxliu	
public:
	cv::Mat get_newK() { return _K_left_new;}
	int undistort_type_{ 3 };
	Mat img_undisted_32;		// 32 bit undistorted image
private:
	string _gammaFile, _vignetteFile; // dir of photometric calibration files
	string _gammaFile2, _vignetteFile2; // dir of photometric calibration files
	float* G;					// photometric result
	float* vignetteMapInv;
	float* _out_data_tmp;

	float* G2;					// photometric result
	float* vignetteMapInv2;
	float* _out_data_tmp2;
	void photometric_init();
	void photometric_init2();
	//void photometric_undistort_parallel(unsigned char* in_data, float* out_data);
	void photometric_undistort_parallel(unsigned char* in_data, float* out_data, int cam_index);
// the end.

private:
	cv::Mat _K_left, _D_left, _K_right, _D_right, _extrinsic_R, _extrinsic_T;
	cv::Mat Q, R1, R2; // don't need to care

	int w, h, wh;					// image width and height 

	Mat rmap[2][2];
	void geometric_init();

public:
	cv::Mat _K_left_new, _K_right_new;		// intrinsic matrix
/*	undist_lite(const int ImgWidth, const int ImgHeight, 
						cv::Mat K_left, cv::Mat D_left, cv::Mat K_right, cv::Mat D_right, 
						cv::Mat extrinsic_R, cv::Mat extrinsic_T);
*/
	//undist_lite(const int ImgWidth, const int ImgHeight, const string gammaFile, const string vignetteFile, cv::Mat K_origin, cv::Mat D, int undistort_type);

	// 2023-2-3
	undist_lite(const int ImgWidth, const int ImgHeight, 
						cv::Mat K_left, cv::Mat D_left, cv::Mat K_right, cv::Mat D_right, 
						cv::Mat extrinsic_R, cv::Mat extrinsic_T, 
						const string gammaFile1, const string vignetteFile1, const string gammaFile2, const string vignetteFile2);
						//const string gammaFile, const string vignetteFile, int undistort_type);
	// the end.

	~undist_lite();
	void undist(cv::Mat &in_img, cv::Mat &out_img);        // include photometric_undistort and geometric_undistort (both parallel)
	void geometric_undistort_parallel(float* in_data, float* out_data);

};

/*
undist_lite::undist_lite(const int ImgWidth, const int ImgHeight, 
						cv::Mat K_left, cv::Mat D_left, cv::Mat K_right, cv::Mat D_right, 
						cv::Mat extrinsic_R, cv::Mat extrinsic_T)
*/
undist_lite::undist_lite(const int ImgWidth, const int ImgHeight, 
						cv::Mat K_left, cv::Mat D_left, cv::Mat K_right, cv::Mat D_right, 
						cv::Mat extrinsic_R, cv::Mat extrinsic_T, 
						const string gammaFile1, const string vignetteFile1, const string gammaFile2, const string vignetteFile2)						
{
	w = ImgWidth, h = ImgHeight, wh = w*h;


	_K_left = K_left;
	_D_left = D_left;
	_K_right = K_right;
	_D_right = D_right;
	_extrinsic_R = extrinsic_R;
	_extrinsic_T = extrinsic_T;

	geometric_init();


	if(sys_cfg_.photometric_cali)
	{
	// added on 2023-2-3
	//undistort_type_ = undistort_type;
	G = new float[256];
	vignetteMapInv = new float[wh];
	_gammaFile = gammaFile1;
	_vignetteFile = vignetteFile1;
	_out_data_tmp = new float[wh];


	G2 = new float[256];
	vignetteMapInv2 = new float[wh];
	_gammaFile2 = gammaFile2;
	_vignetteFile2 = vignetteFile2;
	_out_data_tmp2 = new float[wh];

	img_undisted_32.create(w, h, CV_32FC1);

	photometric_init();
	photometric_init2();
	// the end.
	}
}

undist_lite::~undist_lite()
{
}


void undist_lite::geometric_init()
{
	const double alpha = 0; // mode 1 with black pixel on the egde of the image and mode 0 without 

    cv::Size imageSize(w, h);

    cv::stereoRectify(_K_left, _D_left, _K_right, _D_right, imageSize, 
                      _extrinsic_R, _extrinsic_T,
                      R1,R2,_K_left_new,_K_right_new,Q,
                      cv::CALIB_ZERO_DISPARITY,0,imageSize);//,CALIB_ZERO_DISPARITY,0);

    initUndistortRectifyMap(_K_left,  _D_left,  R1, _K_left_new,  imageSize, CV_32FC1, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(_K_right, _D_right, R2, _K_right_new, imageSize, CV_32FC1, rmap[1][0], rmap[1][1]);
    
    std::cout << "new K_left:"  << std::endl << _K_left_new  << std::endl
              << "new K_right:" << std::endl << _K_right_new << std::endl;
}

void undist_lite::undist(cv::Mat &in_image_l, cv::Mat &in_image_r)
{
/*
	// 2023-2-14
	if(sys_cfg_.photometric_cali) // enable photometric calibration
	{
		photometric_undistort_parallel(in_image_l.data, _out_data_tmp, 1);
		img_undisted_32 = Mat(cv::Size(w, h), CV_32FC1, _out_data_tmp);
		img_undisted_32.convertTo(in_image_l, CV_8UC1);

		{
			photometric_undistort_parallel(in_image_r.data, _out_data_tmp, 2);
			img_undisted_32 = Mat(cv::Size(w, h), CV_32FC1, _out_data_tmp);
			img_undisted_32.convertTo(in_image_r, CV_8UC1);
		}
	}
	// the end.
*/

	remap(in_image_l, in_image_l, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);//左校正
    remap(in_image_r, in_image_r, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);//右校正

	// 2023-2-3
	if(sys_cfg_.photometric_cali) // enable photometric calibration
	{
		photometric_undistort_parallel(in_image_l.data, _out_data_tmp, 1);
		img_undisted_32 = Mat(cv::Size(w, h), CV_32FC1, _out_data_tmp);
		img_undisted_32.convertTo(in_image_l, CV_8UC1);

		{
			photometric_undistort_parallel(in_image_r.data, _out_data_tmp, 2);
			img_undisted_32 = Mat(cv::Size(w, h), CV_32FC1, _out_data_tmp);
			img_undisted_32.convertTo(in_image_r, CV_8UC1);
		}
	}
	// the end.
/**/	
}

// 2023-2-3
/*
void undist_lite::undist(cv::Mat &in_img, cv::Mat &out_img)
{
	photometric_undistort_parallel(in_img.data, _out_data_tmp);

	if(undistort_type_ == 1)
	{
		img_undisted_32 = Mat(cv::Size(w, h), CV_32FC1, _out_data_tmp);
		img_undisted_32.convertTo(out_img, CV_8UC1);
		return ;
	}

	geometric_undistort_parallel(_out_data_tmp, _out_data_32);
	img_undisted_32 = Mat(cv::Size(w, h), CV_32FC1, _out_data_32);

	img_undisted_32.convertTo(out_img, CV_8UC1);
}
*/

void undist_lite::photometric_init()
{
	// read Gamma =================
    std::ifstream f(_gammaFile.data());
	printf("Reading gamma Calibration from file %s\n",_gammaFile.data());
	if (!f.good()){
		printf("Could not open gamma file!\n");
		return;
	}
	// read Vignette ==============
    cv::Mat m = cv::imread(_vignetteFile, cv::IMREAD_UNCHANGED);
	// cv::Mat m = cv::imread("/home/firefly/Documents/OpenCL-Benchmark/undist_lite/calib/vignette.png", cv::IMREAD_UNCHANGED);
	printf("Reading vignette Calibration from file %s\n",_vignetteFile.data());
    if(m.rows*m.cols != wh)
	{
		printf("cv::imread could not read vignette image, size error!!! \n");
		return;
	}
	if(m.type() != CV_16U)
	{
		printf("vignette should be 16bit!!! \n");
		return;
	}
    // process gamma ==============

    std::string line;
	std::getline( f, line );
	std::istringstream l1i( line );
	std::vector<float> Gvec = std::vector<float>( std::istream_iterator<float>(l1i), std::istream_iterator<float>() ); //这玩意就是读出来的gamma向量

    int GDepth = Gvec.size();

	// 检查gamma列数是否大于256
    if(GDepth < 256)
    {
        printf("PhotometricUndistorter: invalid format! got %d entries in first line, expected at least 256!\n",(int)Gvec.size());
        return;
    }

	// 将gamma赋值给G[i]
    for(int i=0;i<GDepth;i++) G[i] = Gvec[i];

	// // 检查gamma是不是单调递增的
    // for(int i=0;i<GDepth-1;i++)
	// {
	// 	if(G[i+1] <= G[i])
	// 	{
	// 		printf("PhotometricUndistorter: G invalid! it has to be strictly increasing, but it isnt!\n");
	// 			return;
	// 	}
	// }
	
	float min=G[0];
    float max=G[GDepth-1];
    for(int i=0;i<GDepth;i++) G[i] = 255.0 * (G[i] - min) / (max-min);		


    // process vignette ===========
    unsigned short* v_data = new unsigned short[m.rows*m.cols];
    memcpy(v_data, m.data, 2*m.rows*m.cols);


    float maxV=0;
	for(int i=0; i<wh; i++)
		if(v_data[i] > maxV) maxV = v_data[i];  // 统计最大像素值

	float* vignetteMap = new float[wh];
	for(int i=0;i<wh;i++)
		vignetteMap[i] = v_data[i] / maxV; // 又归一化??

    if(v_data!=0) delete v_data;

    for(int i=0;i<wh;i++)
	vignetteMapInv[i] = 1.0f / vignetteMap[i]; // 非常朴素的求逆

}

void undist_lite::photometric_init2()
{
	// read Gamma =================
    std::ifstream f(_gammaFile2.data());
	printf("Reading gamma Calibration from file %s\n",_gammaFile2.data());
	if (!f.good()){
		printf("Could not open gamma file!\n");
		return;
	}
	// read Vignette ==============
    cv::Mat m = cv::imread(_vignetteFile2, cv::IMREAD_UNCHANGED);
	// cv::Mat m = cv::imread("/home/firefly/Documents/OpenCL-Benchmark/undist_lite/calib/vignette.png", cv::IMREAD_UNCHANGED);
	printf("Reading vignette Calibration from file %s\n",_vignetteFile2.data());
    if(m.rows*m.cols != wh)
	{
		printf("cv::imread could not read vignette image, size error!!! \n");
		return;
	}
	if(m.type() != CV_16U)
	{
		printf("vignette should be 16bit!!! \n");
		return;
	}
    // process gamma ==============

    std::string line;
	std::getline( f, line );
	std::istringstream l1i( line );
	std::vector<float> Gvec = std::vector<float>( std::istream_iterator<float>(l1i), std::istream_iterator<float>() ); //这玩意就是读出来的gamma向量

    int GDepth = Gvec.size();

	// 检查gamma列数是否大于256
    if(GDepth < 256)
    {
        printf("PhotometricUndistorter: invalid format! got %d entries in first line, expected at least 256!\n",(int)Gvec.size());
        return;
    }

	// 将gamma赋值给G[i]
    for(int i=0;i<GDepth;i++) G2[i] = Gvec[i];

	// // 检查gamma是不是单调递增的
    // for(int i=0;i<GDepth-1;i++)
	// {
	// 	if(G[i+1] <= G[i])
	// 	{
	// 		printf("PhotometricUndistorter: G invalid! it has to be strictly increasing, but it isnt!\n");
	// 			return;
	// 	}
	// }
	
	float min=G2[0];
    float max=G2[GDepth-1];
    for(int i=0;i<GDepth;i++) G2[i] = 255.0 * (G2[i] - min) / (max-min);		


    // process vignette ===========
    unsigned short* v_data = new unsigned short[m.rows*m.cols];
    memcpy(v_data, m.data, 2*m.rows*m.cols);


    float maxV=0;
	for(int i=0; i<wh; i++)
		if(v_data[i] > maxV) maxV = v_data[i];  // 统计最大像素值

	float* vignetteMap = new float[wh];
	for(int i=0;i<wh;i++)
		vignetteMap[i] = v_data[i] / maxV; // 又归一化??

    if(v_data!=0) delete v_data;

    for(int i=0;i<wh;i++)
	vignetteMapInv2[i] = 1.0f / vignetteMap[i]; // 非常朴素的求逆

}

void undist_lite::photometric_undistort_parallel(unsigned char* in_data, float* out_data, int cam_index) // float(&G)[256*256]
{
//#if _USE_PHOTOMETRIC_CALIBRATION == 1
if(cam_index == 1)//(undistort_type_ != 2)
{	
#ifdef __ARM__

    parallel_for_(Range(0, wh), [&](const Range &range)
	{
		for (int idx = range.start; idx < range.end; idx++)
		{
            out_data[idx] = G[in_data[idx]] * vignetteMapInv[idx];
        }
    });

#else

	//const int gsize = 1;
	ParallelFor({0, wh, _WX_GRAIN_SIZE_},
				[&](int idx) { // lambda表达式
					out_data[idx] = G[in_data[idx]] * vignetteMapInv[idx];
				}
	);

#endif
}
else if(cam_index == 2)//(undistort_type_ != 2)
{	
#ifdef __ARM__

    parallel_for_(Range(0, wh), [&](const Range &range)
	{
		for (int idx = range.start; idx < range.end; idx++)
		{
            out_data[idx] = G2[in_data[idx]] * vignetteMapInv2[idx];
        }
    });

#else

	//const int gsize = 1;
	ParallelFor({0, wh, _WX_GRAIN_SIZE_},
				[&](int idx) { // lambda表达式
					out_data[idx] = G2[in_data[idx]] * vignetteMapInv2[idx];
				}
	);

#endif
}
//#else
else
{

#ifdef __ARM__

    parallel_for_(Range(0, wh), [&](const Range &range)
	{
		for (int idx = range.start; idx < range.end; idx++)
		{
            out_data[idx] = in_data[idx];
        }
    });

#else

	//const int gsize = 1;
	ParallelFor({0, wh, _WX_GRAIN_SIZE_},
				[&](int idx) { // lambda表达式
					out_data[idx] = in_data[idx];
				}
	);

#endif
}
//#endif

}
// the end.
