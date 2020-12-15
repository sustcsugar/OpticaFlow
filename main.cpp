#include <stdio.h>  
#include <iostream>  
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include  <opencv2/legacy/legacy.hpp>
#include "opencv2/nonfree/features2d.hpp"   //SurfFeatureDetector实际在该头文件中  
#include <math.h>
//#include <algorithm>
#include <fstream>
//add test comment
using namespace cv;
using namespace std;

//video:720*480
#define CSIZE 4
#define XSIZE 720		//180
#define YSIZE 480		//120
Mat opticalFlow(Mat img1, Mat img2);


int main(int argc, char *argv[])
{
	cout << "argv[0]" << argv[0] << endl;
	cout << "argv[1]" << argv[1] << endl;
	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " YOUR_VIDEO.EXT" << std::endl;
		return 1;
	}
	cout << "argv[0]" << argv[0] << endl;
	cout << "argv[1]" << argv[1] << endl;

	try {
		setNumThreads(1);
		//视频
		VideoCapture videoSource;
		if (!videoSource.open(argv[1])) {
			cout << "ERROR on load video..." << endl;
			return 0;
		}
		Mat frame;
		videoSource.set(CV_CAP_PROP_CONVERT_RGB, 0);
		videoSource >> frame;
		
		Mat noised_dly = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat gray = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat noised = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered2 = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_mb = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_bf = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_bf2 = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_dly = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered2_dly = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat noised_mb = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat noised_dly_mb = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat choose = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat noise = Mat(frame.size(), CV_16S);
		Mat show_channels[3];
		Mat speed_pixel;
		Mat noised_bf;
		Mat noised_forin = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));

		//Sobel参数声明
		Mat grad_x = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat grad_y = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat abs_grad_x = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat abs_grad_y = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat sobel_noised = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat sobel_dly = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		
		int ksize = 1;
		int scale = 1;
		int delta = 0;
		int ddepth = CV_8S;


		int frameCount = 0;
		float noiseStd = 10;
		
		//存储视频用
		int width = frame.cols;
		int height = frame.rows;
		double fps = 30;
		char name[] = "Filtered.avi";
		char name2[] = "Motion.avi";
		char name3[] = "Choosing.avi";
		char name4[] = "sobel.avi";

		VideoWriter Filtered(name, -1, fps, Size(width, height));
		VideoWriter Motion(name2, -1, fps, Size(width, height));
		VideoWriter Choosing(name3, -1, fps, Size(width, height));
		VideoWriter SobelImage(name4, -1, fps, Size(width, height));

		//初始化
		cvtColor(frame, gray, CV_RGB2GRAY);	//转换成灰度图像
		filtered_dly = gray.clone();

		while (1) {
			///*！————————————加噪声——————————————————————！
			imshow("Original", frame);
			cvtColor(frame, gray, CV_RGB2GRAY);	//转换成灰度图像
			imshow("Gray Frame", gray);
			randn(noise, 0, noiseStd);
			add(gray, noise, noised, noArray(), CV_8U);
			imshow("Noised", noised);

			////sobel提取两张图象的边缘
			////noised
			//Sobel(noised, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
			//Sobel(noised, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
			//convertScaleAbs(grad_x, abs_grad_x);
			//convertScaleAbs(grad_y, abs_grad_y);
			//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel_noised);
			//imshow("sobel_noised",sobel_noised);

			////filtered_dly进行sobel变换
			//Sobel(filtered_dly, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
			//Sobel(filtered_dly, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
			//convertScaleAbs(grad_x, abs_grad_x);
			//convertScaleAbs(grad_y, abs_grad_y);
			//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel_dly);

			Mat show = opticalFlow(noised, filtered_dly);
			//提取速度分量
			split(show, show_channels);
			speed_pixel = show_channels[0];


			imshow("speed(sobel)", speed_pixel);
			

			int cols = frame.cols;
			int rows = frame.rows;

			//处理
			//medianBlur(noised, noised_mb, 3);
			//blur(noised, noised_mb, Size(3, 3));
			bilateralFilter(noised, noised_bf, 3, 40, 40);
			filtered_bf = filtered_dly * 0.7 + noised_bf * 0.3;
			filtered_bf2 = filtered_dly * 0.3 + noised_bf * 0.7;
			imshow("bf", noised_bf);

			//遍历速度函数，合成输出图像
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					//判断速度阈值，选择合适的数据源
					if (*speed_pixel.ptr(i, j) <= 60) {
						//cout << *speed_pixel.ptr(i, j) << endl;
						*filtered.ptr(i, j) = *filtered_bf.ptr(i, j);
						*choose.ptr(i, j) = 0;
					}
					else {
						*filtered.ptr(i, j) = *filtered_bf2.ptr(i, j);
						*choose.ptr(i, j) = 255;
					}
				}
			}


			imshow("Filtered", filtered);
			imshow("Choose", choose);

			//前一帧
			filtered_dly = filtered.clone();


			//！————————————保存视频————————————！
			Filtered << filtered;
			Motion << speed_pixel;
			Choosing << choose;
			//SobelImage << sobel_noised;

			//循环关键，不动
			waitKey(1);
			videoSource >> frame;

			frameCount++;
			if (frame.empty()) {
				cout << endl << "Video ended!" << endl;
				break;
			}
		}
	}
	catch (const Exception& ex) {
		cout << "Error: " << ex.what() << endl;
	}
	return 0;
}

/*4x4代码
Mat opticalFlow(Mat img1, Mat img2) {
	img1.convertTo(img1, CV_32F);
	img2.convertTo(img2, CV_32F);
	int i, j, m, n;
	int Ix, Iy, It;
	float Ixx, Iyy, Ixy, Ixt, Iyt;
	float sxx, syy, sxy, sxt, syt;
	float delt1;
	float velx[XSIZE][YSIZE];
	float vely[XSIZE][YSIZE];
	int const row = img2.rows;
	int const col = img2.cols;
	for (i = CSIZE / 2; i <= col - CSIZE / 2; i += CSIZE)
	{
		for (j = CSIZE / 2; j <= row - CSIZE / 2; j += CSIZE)
		{
			int y = (int)(j / CSIZE);
			int x = (int)(i / CSIZE);
			sxx = 0; sxy = 0; syy = 0; sxt = 0; syt = 0;

			for (m = -CSIZE / 2; m < CSIZE / 2; m++)
				for (n = -CSIZE / 2; n < CSIZE / 2; n++)
				{

					if ((i + m + 2) < col && (i + m - 2) > 0) {
						Ix = (-img2.at<float>((j + n), (i + m + 2)) + img2.at<float>((j + n), (i + m + 1)) * 8 - img2.at<float>((j + n), (i + m - 1)) * 8 + img2.at<float>((j + n), (i + m - 2))) / 12;
					}

					else
						Ix = 0;

					if ((j + n + 2) < row && (j + n - 2) > 0)
						Iy = (-img2.at<float>((j + n + 2), (i + m)) + img2.at<float>((j + n + 1), (i + m)) * 8 - img2.at<float>((j + n - 1), (i + m)) * 8 + img2.at<float>((j + n - 2), (i + m))) / 12;
					else
						Iy = 0;

					It = img2.at<float>((j + n), (i + m)) - img1.at<float>((j + n), (i + m));

					Ixx = (float)(Ix * Ix);
					Ixy = (float)(Ix * Iy);
					Iyy = (float)(Iy * Iy);
					Ixt = (float)(Ix * It);
					Iyt = (float)(Iy * It);
					sxx = sxx + Ixx;
					sxy = sxy + Ixy;
					syy = syy + Iyy;
					sxt = sxt + Ixt;
					syt = syt + Iyt;
				}
			delt1 = sxx * syy - sxy * sxy;
			if (delt1 != 0.0)
			{
				velx[x][y] = -(syy * sxt - sxy * syt) / delt1;
				vely[x][y] = -(sxx * syt - sxy * sxt) / delt1;

			}
			else
			{
				velx[x][y] = 0;
				vely[x][y] = 0;
			}
		}
	}
	float d1, d2;
	float max = 0;
	double speedmax[XSIZE][YSIZE];
	for (int i = 0; i < XSIZE - 0; i++)
	{
		for (int j = 0; j < YSIZE - 0; j++) {
			if (i > 0 && j > 0 && i < XSIZE - 1 && j < YSIZE - 1) {

				d1 = ((velx)[i - 1][j - 1] + (velx)[i - 1][j] + (velx)[i - 1][j + 1]
					+ (velx)[i][j - 1] + (velx)[i][j] + (velx)[i][j + 1]
					+ (velx)[i + 1][j - 1] + (velx)[i + 1][j] + (velx)[i + 1][j + 1]) / 9;

				d2 = ((vely)[i - 1][j - 1] + (vely)[i - 1][j] + (vely)[i - 1][j + 1]
					+ (vely)[i][j - 1] + (vely)[i][j] + (vely)[i][j + 1]
					+ (vely)[i + 1][j - 1] + (vely)[i + 1][j] + (vely)[i + 1][j + 1]) / 9;
			}
			else {
				d1 = (velx)[i][j];
				d2 = (vely)[i][j];
			}
			double speed = sqrt(d1 * d1 + d2 * d2);
			speedmax[i][j] = speed;
			if ((max < speed) && (speed < 10)) { 
				max = speed; 
			}
		}
	}
	//cout << max << endl;
	cv::Mat show(row, col, CV_8UC3);
	show = cv::Mat::zeros(Size(col, row), CV_8UC3);


	for (int i = 0; i < XSIZE - 0; i++)
	{
		for (int j = 0; j < YSIZE - 0; j++)
		{
			int y = j * CSIZE;
			int x = i * CSIZE;
			if (i > 0 && j > 0 && i < XSIZE - 1 && j < YSIZE - 1) {

				d1 = ((velx)[i - 1][j - 1] + (velx)[i - 1][j] + (velx)[i - 1][j + 1]
					+ (velx)[i][j - 1] + (velx)[i][j] + (velx)[i][j + 1]
					+ (velx)[i + 1][j - 1] + (velx)[i + 1][j] + (velx)[i + 1][j + 1]) / 9;
				//d1 = velx[i][j];

				d2 = ((vely)[i - 1][j - 1] + (vely)[i - 1][j] + (vely)[i - 1][j + 1]
					+ (vely)[i][j - 1] + (vely)[i][j] + (vely)[i][j + 1]
					+ (vely)[i + 1][j - 1] + (vely)[i + 1][j] + (vely)[i + 1][j + 1]) / 9;
				//d2 = vely[i][j];
			}
			else {
				d1 = (velx)[i][j];
				d2 = (vely)[i][j];
			}
			double speed = sqrt(d1 * d1 + d2 * d2);
			if (i > 0 && j > 0 && i < XSIZE - 0 && j < YSIZE - 0) {
				for (int u = 0; u < CSIZE; u++)
					for (int v = 1; v <= CSIZE; v++) {
						show.at<Vec3b>(y - v, x - u) = Vec3b(int(255 * speed / max), int(255 - 255 * speed / max), 0);
					}
			}
		}
	}
	return show;
}

*/


///*1x1代码

Mat opticalFlow(Mat img1, Mat img2) {
	img1.convertTo(img1, CV_32F);
	img2.convertTo(img2, CV_32F);
	int i, j, m, n;
	int Ix, Iy, It;
	float Ixx, Iyy, Ixy, Ixt, Iyt;
	float sxx, syy, sxy, sxt, syt;
	float delt1;
	float velx[XSIZE][YSIZE];
	float vely[XSIZE][YSIZE];
	int const row = img2.rows;
	int const col = img2.cols;
	float delt1out[XSIZE][YSIZE];
	float syysxt[XSIZE][YSIZE];
	float sxysyt[XSIZE][YSIZE];
	float sxxsyt[XSIZE][YSIZE];
	float sxysxt[XSIZE][YSIZE];

	for (i = CSIZE / 2; i <= col - CSIZE / 2; i += 1)//CSIZE
		for (j = CSIZE / 2; j <= row - CSIZE / 2; j += 1)//CSIZE
		{
			int y = (int)(j / 1);//CSIZE
			int x = (int)(i / 1);//CSIZE
			sxx = 0; sxy = 0; syy = 0; sxt = 0; syt = 0;

			for (m = -CSIZE / 2; m < CSIZE / 2; m++)
				for (n = -CSIZE / 2; n < CSIZE / 2; n++)
				{

					if ((i + m + 2) < col && (i + m - 2) > 0) {
						Ix = (-img2.at<float>((j + n), (i + m + 2)) + img2.at<float>((j + n), (i + m + 1)) * 8 - img2.at<float>((j + n), (i + m - 1)) * 8 + img2.at<float>((j + n), (i + m - 2))) / 12;
					}

					else
						Ix = 0;

					if ((j + n + 2) < row && (j + n - 2) > 0)
						Iy = (-img2.at<float>((j + n + 2), (i + m)) + img2.at<float>((j + n + 1), (i + m)) * 8 - img2.at<float>((j + n - 1), (i + m)) * 8 + img2.at<float>((j + n - 2), (i + m))) / 12;
					else
						Iy = 0;

					It = img2.at<float>((j + n), (i + m)) - img1.at<float>((j + n), (i + m));

					Ixx = (float)(Ix * Ix);
					Ixy = (float)(Ix * Iy);
					Iyy = (float)(Iy * Iy);
					Ixt = (float)(Ix * It);
					Iyt = (float)(Iy * It);
					sxx = sxx + Ixx;
					sxy = sxy + Ixy;
					syy = syy + Iyy;
					sxt = sxt + Ixt;
					syt = syt + Iyt;
				}
			delt1 = sxx * syy - sxy * sxy;

			if (delt1 != 0.0)
			{
				velx[x][y] = -(syy * sxt - sxy * syt) / delt1;
				vely[x][y] = -(sxx * syt - sxy * sxt) / delt1;
				syysxt[x][y] = syy * sxt;
				sxysyt[x][y] = sxy * syt;
				sxxsyt[x][y] = sxx * syt;
				sxysxt[x][y] = sxy * sxt;

			}
			else
			{
				velx[x][y] = 0;
				vely[x][y] = 0;

			}

		}
	float d1, d2;
	float max = 0;
	double speedmax[XSIZE][YSIZE];
	for (int i = 0; i < XSIZE - 0; i++)
		for (int j = 0; j < YSIZE - 0; j++) {
			if (i > 0 && j > 0 && i < XSIZE - 1 && j < YSIZE - 1) {

				d1 = ((velx)[i - 1][j - 1] + (velx)[i - 1][j] + (velx)[i - 1][j + 1]
					+ (velx)[i][j - 1] + (velx)[i][j] + (velx)[i][j + 1]
					+ (velx)[i + 1][j - 1] + (velx)[i + 1][j] + (velx)[i + 1][j + 1]) / 9;

				d2 = ((vely)[i - 1][j - 1] + (vely)[i - 1][j] + (vely)[i - 1][j + 1]
					+ (vely)[i][j - 1] + (vely)[i][j] + (vely)[i][j + 1]
					+ (vely)[i + 1][j - 1] + (vely)[i + 1][j] + (vely)[i + 1][j + 1]) / 9;
			}
			else {
				d1 = (velx)[i][j];
				d2 = (vely)[i][j];
			}
			double speed = sqrt(d1 * d1 + d2 * d2);
			speedmax[i][j] = speed;
			if ((max < speed) && (speed < 10)) { max = speed; }
		}
	cout << max << endl;
	cv::Mat show(row, col, CV_8UC3);



	for (int i = 0; i < XSIZE - 0; i++) {
		for (int j = 0; j < YSIZE - 0; j++) {
			int y = j * 1;//CSIZE
			int x = i * 1;//CSIZE
			if (i > 0 && j > 0 && i < XSIZE - 1 && j < YSIZE - 1) {

				d1 = ((velx)[i - 1][j - 1] + (velx)[i - 1][j] + (velx)[i - 1][j + 1]
					+ (velx)[i][j - 1] + (velx)[i][j] + (velx)[i][j + 1]
					+ (velx)[i + 1][j - 1] + (velx)[i + 1][j] + (velx)[i + 1][j + 1]) / 9;

				d2 = ((vely)[i - 1][j - 1] + (vely)[i - 1][j] + (vely)[i - 1][j + 1]
					+ (vely)[i][j - 1] + (vely)[i][j] + (vely)[i][j + 1]
					+ (vely)[i + 1][j - 1] + (vely)[i + 1][j] + (vely)[i + 1][j + 1]) / 9;
			}
			else {
				d1 = (velx)[i][j];
				d2 = (vely)[i][j];
			}
			double speed = sqrt(d1 * d1 + d2 * d2);
			if (i > 0 && j > 0 && i <= XSIZE - 0 && j <= YSIZE - 0) {
				//				for (int u = 0; u < CSIZE; u++)
				//					for (int v = 1; v <= CSIZE; v++) {
				show.at<Vec3b>(y, x) = Vec3b(int(255 * speed / max), int(255 - 255 * speed / max), 0);

				//						show.at<Vec3b>(y - v, x - u) = Vec3b(int(255 - 255 * speed / max), int(255 - 255 * speed / max), 0);
				//					}
			}
		}
	}

	//imshow("optical_flowflow", show);
	//imwrite("optical_flow.png", img);
	//cv::waitKey(0);
	return show;
}
