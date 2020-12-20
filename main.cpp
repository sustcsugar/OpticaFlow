#include <stdio.h>  
#include <iostream>  
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include  <opencv2/legacy/legacy.hpp>
#include "opencv2/nonfree/features2d.hpp"   //SurfFeatureDetector实际在该头文件中  
#include <math.h>

//金字塔光流调用
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <fstream>

#define ATD at<double>


using namespace cv;
using namespace std;

//video:720*480
#define CSIZE 4
#define XSIZE 720//180		
#define YSIZE 480//120	

Mat opticalFlow(Mat img1, Mat img2);
Mat opticalFlow_pyramid(Mat& img1, Mat& img2, Mat& vel_up, int CSIZE_ALL);
int max(double a, double b);


int main(int argc, char *argv[])
{
	cout << "argv[0]" << argv[0] << endl;
	cout << "argv[1]" << argv[1] << endl;
	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " YOUR_VIDEO.EXT" << std::endl;
		return 1;
	}

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
		
		//调整图像尺寸
		//resize(frame, frame, Size(640, 480), 0, 0, INTER_AREA);

		//图像矩阵声明
		Mat noised_dly		= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat gray			= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat noised			= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered		= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered2		= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_mb		= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_bf		= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_bf2	= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_bf82	= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_bf73	= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_bf64	= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_bf55	= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_bf46	= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_bf37	= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_bf28	= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered_dly	= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat filtered2_dly	= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));

		Mat noised_mb		= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat noised_dly_mb	= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
		Mat choose			= Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));


		Mat noise			= Mat(frame.size(), CV_16S);
		Mat show_channels[3];
		Mat speed_pixel;
		Mat noised_bf;
		Mat noised_bf_dly;
		Mat noised_forin = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));

		int frameCount = 0;
		float noiseStd = 10;	//生成noise的参数
		
		//写入视频用
		int width = frame.cols;
		int height = frame.rows;
		double fps = 30;
		char name[] = "Filtered.avi";
		char name2[] = "Motion.avi";
		char name3[] = "Choosing.avi";

		VideoWriter Filtered(name, -1, fps, Size(width, height));
		VideoWriter Motion(name2, -1, fps, Size(width, height));
		VideoWriter Choosing(name3, -1, fps, Size(width, height));
		//VideoWriter OP(name4, -1, fps, Size(width, height));

		
		//存储文件用
		ofstream ofile;
		ofile.open("D:\\Desktop\\OptialFlow\\speed.txt");

		//初始化
		cvtColor(frame, gray, CV_RGB2GRAY);	//转换成灰度图像
		//imshow("Gray Frame", gray);
		//filtered_dly = gray.clone();
		//noised_bf_dly = gray.clone();


		while (1) {
			//！————————————加噪声——————————————————————！
			imshow("Original", frame);
			cvtColor(frame, gray, CV_RGB2GRAY);	//转换成灰度图像
			imshow("Gray Frame", gray);
			randn(noise, 0, noiseStd);
			add(gray, noise, noised, noArray(), CV_8U);
			imshow("Noised", noised);

			//图像预处理
			//medianBlur(noised, noised_mb, 3);
			//blur(noised, noised_mb, Size(3, 3));
			bilateralFilter(noised, noised_bf, 3, 40, 40);
			imshow("noised_bf", noised_bf);
			//filtered_bf55 = filtered_dly * 0.5 + noised_bf * 0, 5;


			////Sobel求轮廓
			////求x方向梯度——noised_bf
			//Sobel(noised_bf, grad_x1, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
			//convertScaleAbs(grad_x1, abs_grad_x1);
			////求y方向梯度
			//Sobel(noised_bf, grad_y1, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
			//convertScaleAbs(grad_y1, abs_grad_y1);
			////合并梯度
			//addWeighted(abs_grad_x1, 0.5, abs_grad_y1, 0.5, 0, dst1);
			//imshow("Sobel算法轮廓提取效果nosied", dst1);

			////求x方向梯度——filtered_dly
			//Sobel(noised_bf_dly, grad_x2, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
			//convertScaleAbs(grad_x2, abs_grad_x2);
			////求y方向梯度
			//Sobel(noised_bf_dly, grad_y2, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
			//convertScaleAbs(grad_y2, abs_grad_y2);
			////合并梯度
			//addWeighted(abs_grad_x2, 0.5, abs_grad_y2, 0.5, 0, dst2);
			//imshow("Sobel算法轮廓提取效果filtered", dst2);
			


			//noised_bf_dly = noised_bf.clone();


			//光流法
			//Mat show = opticalFlow(dst1, dst2);
			Mat show = opticalFlow(filtered_bf55, noised_bf);

			////金字塔光流
			//	//声明
			//Mat img1_p0 = filtered_dly;
			//Mat img2_p0 = noised_bf;
			//resize(img1_p0, img1_p0, Size(640, 480), 0, 0, INTER_AREA);
			//resize(img2_p0, img2_p0, Size(640, 480), 0, 0, INTER_AREA);
			//Mat img1_p1, img1_p2, img2_p1, img2_p2, img2_p3;
			//img1_p1 = img1_p0;
			//img2_p1 = img1_p0;
			////initial img pyr
			//pyrDown(img1_p0, img1_p1, Size(img1_p0.cols / 2, img1_p0.rows / 2));
			//pyrDown(img1_p1, img1_p2, Size(img1_p1.cols / 2, img1_p1.rows / 2));
			//pyrDown(img2_p0, img2_p1, Size(img2_p0.cols / 2, img2_p0.rows / 2));
			//pyrDown(img2_p1, img2_p2, Size(img2_p1.cols / 2, img2_p1.rows / 2));
			//pyrDown(img2_p2, img2_p3, Size(img2_p2.cols / 2, img2_p2.rows / 2));
			//
			//Mat vel_up(img2_p3.size(), CV_32FC2, Scalar(0));
			//opticalFlow_pyramid(img1_p2, img2_p2, vel_up, 4);
			//opticalFlow_pyramid(img1_p1, img2_p1, vel_up, 8);
			//Mat show = opticalFlow_pyramid(img1_p0, img2_p0, vel_up, 16);
			//Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
			//dilate(show, show, element);
			//imshow("pyramid", show);

			
			//提取速度分量
			split(show, show_channels);
			speed_pixel = show_channels[0];
			imshow("speed", speed_pixel);	

			int rows = frame.rows;
			int cols = frame.cols;

			////简单的73和37
			//filtered_bf = filtered_dly * 0.8 + noised_bf * 0.2;
			//filtered_bf2 = filtered_dly * 0.3 + noised_bf * 0.7;

			////根据速度函数，合成输出图像
			//for (int i = 0; i < rows; i++) {
			//	for (int j = 0; j < cols; j++) {
			//		ofile << (int)*speed_pixel.ptr(i, j) << " ";
			//		//判断速度阈值，选择合适的数据源
			//		if (*speed_pixel.ptr(i, j) <= 35) {
			//			*filtered.ptr(i, j) = *filtered_bf.ptr(i, j);
			//			*choose.ptr(i, j) = 0;
			//		}
			//		else {
			//			*filtered.ptr(i, j) = *filtered_bf2.ptr(i, j);
			//			*choose.ptr(i, j) = 255;
			//		}
			//	}
			//	ofile << endl;
			//} 
			
			////扩展区域的合成
			filtered_bf82 = filtered_dly * 0.8 + noised_bf * 0.2;
			filtered_bf73 = filtered_dly * 0.7 + noised_bf * 0.3;
			//filtered_bf64 = filtered_dly * 0.6 + noised_bf * 0.4;
			filtered_bf55 = filtered_dly * 0.5 + noised_bf * 0.5;
			filtered_bf46 = filtered_dly * 0.4 + noised_bf * 0.6;
			filtered_bf37 = filtered_dly * 0.3 + noised_bf * 0.7;
		

			int proportion[750][750];

			////边框默认不运动,金字塔算法中边框的13个像素点是255，这里修改为0
			//for (int i = 0; i < rows; i++) {
			//	for (int j = 0; j < cols; j++) {
			//		if (i <= 13 || j <= 13 || i >= rows - 13 || j >= cols - 13) {
			//			*speed_pixel.ptr(i, j) = 0;
			//		}
			//	}
			//}
			//int pz = 20;//高速度周围膨胀像素点
			//int pz2 = 15;//一般速度周围膨胀像素点

			////*********************静止区域
			//for (int i = 0; i < rows; i++) {
			//	for (int j = 0; j < cols; j++) {
			//		if (*speed_pixel.ptr(i, j) < 30) {		
			//			proportion[i][j] = 0;	//根据这个分配合成比例
			//			*choose.ptr(i, j) = 0;	//choose矩阵用来观察
			//		}
			//	}
			//}

			////******************低速区域块扩展
			////proportion[i][j] = 1;
			////*choose.ptr(i, j) = 80;
			//for (int i = 0; i < rows; i++) {
			//	for (int j = 0; j < cols; j++) {
			//		if (*speed_pixel.ptr(i, j) > 30) { 							
			//			if (i < pz2 && j < pz2) {
			//				for (int k = 0; k < i + pz2; k++) {
			//					for (int z = 0; z < j + pz2; z++) {
			//						proportion[k][z] = 1;
			//						if (k < rows && z < cols) {
			//							*choose.ptr(k, z) = 80;
			//						}
			//					}
			//				}
			//			}
			//			if (i < pz2 && j > pz2) {
			//				for (int k = 0; k < i + pz2; k++) {
			//					for (int z = j - pz2; z < j + pz2; z++) {
			//						proportion[k][z] = 1;
			//						if (k < rows && z < cols) {
			//							*choose.ptr(k, z) = 80;
			//						}
			//					}
			//				}
			//			}
			//			if (i > pz2 && j < pz2) {
			//				for (int k = i - pz2; k < i + pz2; k++) {
			//					for (int z = 0; z < j + pz2; z++) {
			//						proportion[k][z] = 1;
			//						if (k < rows && z < cols) {
			//							*choose.ptr(k, z) = 80;
			//						}
			//					}
			//				}
			//			}
			//			if (i > pz2 && j > pz2) {
			//				for (int k = i - pz2; k <= i + pz2; k++) {
			//					for (int z = j - pz2; z <= j + pz2; z++) {
			//						proportion[k][z] = 1;
			//						if (k < rows && z < cols) {
			//							*choose.ptr(k, z) = 80;
			//						}
			//					}
			//				}
			//			}	
			//		}
			//	}
			//}

			////*******************高速区域块扩展
			////proportion[i][j] = 2;
			////*choose.ptr(i, j) = 160;
			//for (int i = 0; i < rows; i++) {
			//	for (int j = 0; j < cols; j++) {
			//		if (*speed_pixel.ptr(i, j) > 65) {
			//			if (i < pz && j < pz) {
			//				for (int k = 0; k < i + pz; k++) {
			//					for (int z = 0; z < j + pz; z++) {
			//						proportion[k][z] = 2;
			//						if (k < rows && z < cols) {
			//							*choose.ptr(k, z) = 160;
			//						}
			//					}
			//				}
			//			}
			//			if (i < pz && j > pz) {
			//				for (int k = 0; k < i + pz; k++) {
			//					for (int z = j - pz; z < j + pz; z++) {
			//						proportion[k][z] = 2;
			//						if (k < rows && z < cols) {
			//							*choose.ptr(k, z) = 160;
			//						}
			//					}
			//				}
			//			}
			//			if (i > pz && j < pz) {
			//				for (int k = i - pz; k < i + pz; k++) {
			//					for (int z = 0; z < j + pz; z++) {
			//						proportion[k][z] = 2;
			//						if (k < rows && z < cols) {
			//							*choose.ptr(k, z) = 160;
			//						}
			//					}
			//				}
			//			}
			//			if (i >= pz && j >= pz) {
			//				for (int k = i - pz; k <= i + pz; k++) {
			//					for (int z = j - pz; z <= j + pz; z++) {
			//						proportion[k][z] = 2;
			//						if (k < rows && z < cols) {
			//							*choose.ptr(k, z) = 160;
			//						}
			//					}
			//				}
			//			}
			//		}
			//	}
			//}

			////高速区域填回
			////3,155
			//for (int i = 0; i < rows; i++) {
			//	for (int j = 0; j < cols; j++) {
			//		if (*speed_pixel.ptr(i, j) > 65) {
			//			proportion[i][j] = 3;
			//			*choose.ptr(i, j) = 255;
			//		}
			//	}
			//}

			////根据分布图，合成图像
			//for (int i = 0; i < rows; i++) {
			//	for (int j = 0; j < cols; j++) {
			//		if (proportion[i][j] == 0) {
			//			*filtered.ptr(i, j) = *filtered_bf82.ptr(i, j);
			//		}
			//		else if (proportion[i][j] == 1) {
			//			*filtered.ptr(i, j) = *filtered_bf55.ptr(i, j);
			//		}
			//		else if (proportion[i][j] == 2) {
			//			*filtered.ptr(i, j) = *filtered_bf46.ptr(i, j);
			//		}
			//		else {
			//			*filtered.ptr(i, j) = *filtered_bf37.ptr(i, j);
			//		}
			//	}
			//}				







			//wxl*********
			//边框默认不运动,金字塔算法中边框的13个像素点是255，这里修改为0
			///*线性扩展
			//根据速度函数，得到各点图像合成比例图
			//这里的proportion取值从20到0；
			//边框默认不运动,光流法算法中边框的3个像素点是255，这里修改为0
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (i <= 3 || j <= 3 || i >= rows - 3 || j >= cols - 3) {
						*speed_pixel.ptr(i, j) = 0;
					}
				}
			}
			double d_high = 50;//高速区域扩展距离
			double d_low = 30;//低速

			double q_high = 50;//高速区域扩展中心强度
			double q_low = 30;

			//静止区域
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (*speed_pixel.ptr(i, j) < 30) {
						proportion[i][j] = 0;
					}
				}
			}
			//低速周围快扩展
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (*speed_pixel.ptr(i, j) >= 30) {
						//记住，待修改
						for (int k = 0; k < rows; k++) {
							for (int z = 0; z < cols; z++) {
								double distance = sqrt((i - k)*(i - k) + (j - z)*(j - z));
								if (d_low - distance >= 0) {
									proportion[k][z] = max(proportion[k][z], (-(d_low / (q_low*q_low))*distance*distance + q_low));
								}
							}
						}
					}
				}
			}
			//高速周围快扩展
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (*speed_pixel.ptr(i, j) >= 55) {
						for (int k = 0; k < rows; k++) {
							for (int z = 0; z < cols; z++) {
								double distance = sqrt((i - k)*(i - k) + (j - z)*(j - z));
								if (d_high - distance >= 0) {
									proportion[k][z] = max(proportion[k][z], (-(d_high / (q_high*q_high))*distance*distance + q_high));
								}
							}
						}
					}
				}
			}

			//观察用
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					*choose.ptr(i, j) = proportion[i][j] * (255 / q_high);
				}
			}

			//根据分布图，合成图像
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					*filtered.ptr(i, j) = (0.8 - 0.6*(proportion[i][j] / q_high))**filtered_dly.ptr(i, j) + (0.2 + 0.6*(proportion[i][j] / q_high))**noised_bf.ptr(i, j);
				}
			}
			//*/





			
			imshow("Filtered", filtered);
			imshow("Choose", choose);
			//前一帧
			filtered_dly = filtered.clone();
			//filtered_dly = filtered;

			//！————————————保存视频————————————！
			Filtered << filtered;
			Motion << speed_pixel;
			Choosing << choose;

			cout << "当前帧数:" << frameCount << " /162 (5s);" << endl;
			//ofile << endl << endl;

			//循环关键，不动
			waitKey(1);

			videoSource >> frame;
			
			//resize(frame, frame, Size(640, 480), 0, 0, INTER_AREA);

			frameCount++;
			if (frame.empty()) {
				cout << endl << "Video ended!" << endl;
				break;
			}
		}
		ofile.close();
	}
	catch (const Exception& ex) {
		cout << "Error: " << ex.what() << endl;
	}

	return 1;
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


//1x1代码
//需要添加函数的注释,参数说明
//show三个通道的意义
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
			if ((max < speed) && (speed < 10)) { max = speed; } //???
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
				show.at<Vec3b>(y, x) = Vec3b(int(10 * speed), int(255 - 255 * speed / max), 0);

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




//输入img1,img2
//vel_up表示 ?? 
Mat opticalFlow_pyramid(Mat& img1, Mat& img2, Mat& vel_up, int CSIZE_ALL) {
	img1.convertTo(img1, CV_32F);
	img2.convertTo(img2, CV_32F);
	vel_up.convertTo(vel_up, CV_32F);
	ofstream myout("C:/Users/Sugar_desktop/Desktop/bg.txt");

	int i, j, m, n;
	int Ix, Iy, It;
	float Ixx, Iyy, Ixy, Ixt, Iyt;
	float sxx, syy, sxy, sxt, syt;
	float delt1;
	int const row = img2.rows;
	int const col = img2.cols;
	Mat vel(row, col, CV_32FC2);//[1]x,[2]y
	int vx_up, vy_up, velx_up, vely_up;
	Mat show(img2.size(), CV_8UC3);
	int a;
	for (i = CSIZE_ALL / 2; i <= col - CSIZE_ALL / 2; i += 1)//CSIZE
	{
		for (j = CSIZE_ALL / 2; j <= row - CSIZE_ALL / 2; j += 1)//CSIZE
		{
			int y = (int)(j / 1);//CSIZE
			int x = (int)(i / 1);//CSIZE
			sxx = 0; sxy = 0; syy = 0; sxt = 0; syt = 0;

			velx_up = 2 * vel_up.at<Vec2f>(round(y / 2), round(x / 2))[0];
			vely_up = 2 * vel_up.at<Vec2f>(round(y / 2), round(x / 2))[1];

			for (m = -CSIZE / 2; m < CSIZE / 2; m++)
				for (n = -CSIZE / 2; n < CSIZE / 2; n++)
				{
					vx_up = velx_up + m;
					vy_up = vely_up + n;
					if ((vx_up + m) > CSIZE_ALL / 2) { vx_up = CSIZE_ALL / 2 - 1; }
					if ((vx_up + m) < -CSIZE_ALL / 2) { vx_up = -CSIZE_ALL / 2; }
					if ((vy_up + n) > CSIZE_ALL / 2) { vy_up = CSIZE_ALL / 2 - 1; }
					if ((vy_up + n) < -CSIZE_ALL / 2) { vy_up = CSIZE_ALL / 2; }

					if ((i + m + 2) < col && (i + m - 2) > 0 && (j + n + 2) < row && (j + n - 2) > 0) {
						Ix = (-img1.at<float>((j + n), (i + m + 2)) + img1.at<float>((j + n), (i + m + 1)) * 8 - img1.at<float>((j + n), (i + m - 1)) * 8 + img1.at<float>((j + n), (i + m - 2))) / 12;
					}

					else
						Ix = 0;

					if ((i + m + 2) < col && (i + m - 2) > 0 && (j + n + 2) < row && (j + n - 2) > 0)
						Iy = (-img1.at<float>((j + n + 2), (i + m)) + img1.at<float>((j + n + 1), (i + m)) * 8 - img1.at<float>((j + n - 1), (i + m)) * 8 + img1.at<float>((j + n - 2), (i + m))) / 12;
					else
						Iy = 0;
					if ((i + vx_up + 2) < col && (i + vx_up - 2) > 0 && (j + vy_up + 2) < row && (j + vy_up - 2) > 0)
						It = img2.at<float>((j + vy_up), (i + vx_up)) - img1.at<float>((j + vy_up), (i + vx_up));
					else
						It = 0;
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
				vel.at<Vec2f>(y, x)[0] = -(syy * sxt - sxy * syt) / delt1;
				vel.at<Vec2f>(y, x)[1] = -(sxx * syt - sxy * sxt) / delt1;
				//	int b = vel.at<Vec2f>(y, x)[0];
				//	if (b > 1) { cout << b << endl; }
			}
			else
			{
				vel.at<Vec2f>(y, x)[0] = 0;
				vel.at<Vec2f>(y, x)[1] = 0;

			}
			a = int(pow(vel.at<Vec2f>(y, x)[0], 2) + pow(vel.at<Vec2f>(y, x)[1], 2));
			if (a < 0) { a = 0; }
			if (a > 255) { a = 255; }
			myout << a << endl;
			show.at<Vec3b>(y, x) = Vec3b(a, 0, 0);
		}
	}
	vel_up = vel;
	return show;
}


int max(double a, double b)
{
	if (a > b) return a; return b;
}