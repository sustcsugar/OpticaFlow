#include <iostream>
#include "opencv2/opencv.hpp"
#include <math.h>
#include <algorithm>
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#define CSIZE 4
//#define XSIZE 640//160
//#define YSIZE 480//120
#define ATD at<double>

Mat opticalFlow(Mat& img1, Mat& img2, Mat& vel_up, int CSIZE_ALL);

int main(int argc, char *argv[]) {


	//check parameter
	cout << "argv[0]" << argv[0] << endl;
	cout << "argv[1]" << argv[1] << endl;
	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " YOUR_VIDEO.EXT" << std::endl;
		return 1;
	}

	//read video
	VideoCapture videoSource;
	if (!videoSource.open(argv[1])) {
		cout << "ERROR on load video..." << endl;
		return 0;
	}
	Mat frame;
	videoSource.set(CV_CAP_PROP_CONVERT_RGB, 0);
	videoSource >> frame;

	//declare Mat variables
	Mat gray = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
	Mat noise = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
	Mat noised = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
	Mat filtered_dly = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));
	Mat noised_dly = Mat(Size(frame.cols, frame.rows), CV_8UC1, Scalar(255));

	int frameCount = 0;
	float noiseStd = 10;


	//**************************		write video
	//int width = frame.cols;
	//int height = frame.rows;
	//double fps = 30;
	//char name[] = "show.avi";
	//char name2[] = "Motion.avi";
	//char name3[] = "Choosing.avi";

	//VideoWriter Show(name, -1, fps, Size(640, 480));


	//***************************			initialization
	cvtColor(frame, gray, CV_RGB2GRAY);	
	filtered_dly = gray.clone();

	// bilateral parameter
	int d = 5;
	double sigma = 10;

	while (1) {

		//***********************	add noise	*****************
		imshow("Original", frame);
		cvtColor(frame, gray, CV_RGB2GRAY);
		imshow("gray frame", gray);
		randn(noise, 0, noiseStd);
		add(gray, noise,noised,noArray(), CV_8U);
		imshow("noised", noised);
		
		bilateralFilter(noised, noised, d, sigma, sigma);

		//**********************	pyramid motion detection	*******************
		//Mat img = filtered_dly;
		Mat img1_p0 = filtered_dly;
		Mat img2_p0 = noised;
		
		
		//resize(img, img, Size(640, 480), 0, 0, INTER_AREA);
		resize(img1_p0, img1_p0, Size(640, 480), 0, 0, INTER_AREA);
		resize(img2_p0, img2_p0, Size(640, 480), 0, 0, INTER_AREA);

		Mat img1_p1, img1_p2, img2_p1, img2_p2, img2_p3;
		img1_p1 = img1_p0;
		img2_p1 = img2_p0;

		//initial img pyr
		pyrDown(img1_p0, img1_p1, Size(img1_p0.cols / 2, img1_p0.rows / 2));
		pyrDown(img1_p1, img1_p2, Size(img1_p1.cols / 2, img1_p1.rows / 2));
		pyrDown(img2_p0, img2_p1, Size(img2_p0.cols / 2, img2_p0.rows / 2));
		pyrDown(img2_p1, img2_p2, Size(img2_p1.cols / 2, img2_p1.rows / 2));

		Mat show;
		Mat vel_up(img2_p2.size(), CV_32FC2, Scalar(0));
		opticalFlow(img1_p2, img2_p2, vel_up, 4);
		opticalFlow(img1_p1, img2_p1, vel_up, 8);
		show = opticalFlow(img1_p0, img2_p0, vel_up, 16);

		Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));
		dilate(show, show, element);
		Mat show_channels[3];
		Mat speed_pixel;
		split(show, show_channels);
		speed_pixel = show_channels[2];


		imshow("motion", speed_pixel);

		// 保存上一帧数据
		filtered_dly = noised;
		waitKey(1);

		videoSource >> frame;
		frameCount++;

		if (frame.empty()) {
			cout << endl << "Video ended!" << endl;
			break;
		}
	}
	return 0;
}



Mat opticalFlow(Mat& img1, Mat& img2, Mat& vel_up,int CSIZE_ALL) {
	img1.convertTo(img1, CV_32F);
	img2.convertTo(img2, CV_32F);
	vel_up.convertTo(vel_up, CV_32F);
	ofstream myout("D:/Desktop/bg.txt");

	int i, j, m, n;
	int Ix, Iy, It;
	float Ixx, Iyy, Ixy, Ixt, Iyt;
	float sxx, syy, sxy, sxt, syt;
	float delt1;
	int const row = img2.rows;
	int const col = img2.cols;
	Mat vel(row, col, CV_32FC2);//[1]x,[2]y
	int vx_up, vy_up, velx_up,vely_up;
	Mat show (img2.size(), CV_8UC3);
	int a;
	for (i = CSIZE_ALL / 2; i <= col - CSIZE_ALL / 2; i += 1)//CSIZE
	{
		for (j = CSIZE_ALL / 2; j <= row - CSIZE_ALL / 2; j += 1)//CSIZE
		{
			int y = (int)(j / 1);//CSIZE
			int x = (int)(i / 1);//CSIZE
			sxx = 0; sxy = 0; syy = 0; sxt = 0; syt = 0;

			velx_up = 2*vel_up.at<Vec2f>(round(y / 2), round(x / 2))[0];
			vely_up = 2*vel_up.at<Vec2f>(round(y / 2), round(x / 2))[1];

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
			if (a < 0) { a = 0;}
			if (a > 255) {  a = 255;}
			myout <<a<< endl;
			show.at<Vec3b>(y, x) = Vec3b(0, 0, a);
		}
	}
	vel_up = vel;
	return show;
	//float d1, d2;
	//float max = 0;
	//double speedmax[XSIZE][YSIZE];
	//for (int i = 0; i < XSIZE - 0; i++)
	//	for (int j = 0; j < YSIZE - 0; j++) {
	//		if (i > 0 && j > 0 && i < XSIZE - 1 && j < YSIZE - 1) {

	//			d1 = ((velx)[i - 1][j - 1] + (velx)[i - 1][j] + (velx)[i - 1][j + 1]
	//				+ (velx)[i][j - 1] + (velx)[i][j] + (velx)[i][j + 1]
	//				+ (velx)[i + 1][j - 1] + (velx)[i + 1][j] + (velx)[i + 1][j + 1]) / 9;

	//			d2 = ((vely)[i - 1][j - 1] + (vely)[i - 1][j] + (vely)[i - 1][j + 1]
	//				+ (vely)[i][j - 1] + (vely)[i][j] + (vely)[i][j + 1]
	//				+ (vely)[i + 1][j - 1] + (vely)[i + 1][j] + (vely)[i + 1][j + 1]) / 9;
	//		}
	//		else {
	//			d1 = (velx)[i][j];
	//			d2 = (vely)[i][j];
	//		}
	//		double speed = sqrt(d1 * d1 + d2 * d2);
	//		speedmax[i][j] = speed;
	//		if ((max < speed) && (speed < 10)) { max = speed; }
	//	}
	//cout << max << endl;
	//cv::Mat show(row, col, CV_8UC3);



	//for (int i = 0; i < XSIZE - 0; i++) {
	//	for (int j = 0; j < YSIZE - 0; j++) {
	//		int y = j * 1;//CSIZE
	//		int x = i * 1;//CSIZE
	//		if (i > 0 && j > 0 && i < XSIZE - 1 && j < YSIZE - 1) {

	//			d1 = ((velx)[i - 1][j - 1] + (velx)[i - 1][j] + (velx)[i - 1][j + 1]
	//				+ (velx)[i][j - 1] + (velx)[i][j] + (velx)[i][j + 1]
	//				+ (velx)[i + 1][j - 1] + (velx)[i + 1][j] + (velx)[i + 1][j + 1]) / 9;

	//			d2 = ((vely)[i - 1][j - 1] + (vely)[i - 1][j] + (vely)[i - 1][j + 1]
	//				+ (vely)[i][j - 1] + (vely)[i][j] + (vely)[i][j + 1]
	//				+ (vely)[i + 1][j - 1] + (vely)[i + 1][j] + (vely)[i + 1][j + 1]) / 9;
	//		}
	//		else {
	//			d1 = (velx)[i][j];
	//			d2 = (vely)[i][j];
	//		}
	//		double speed = sqrt(d1 * d1 + d2 * d2);
	//		if (i > 0 && j > 0 && i <= XSIZE - 0 && j <= YSIZE - 0) {
	//			//				for (int u = 0; u < CSIZE; u++)
	//			//					for (int v = 1; v <= CSIZE; v++) {
	//			show.at<Vec3b>(y, x) = Vec3b(int(255 * speed / max), int(255 - 255 * speed / max), 0);

	//			//						show.at<Vec3b>(y - v, x - u) = Vec3b(int(255 - 255 * speed / max), int(255 - 255 * speed / max), 0);
	//			//					}
	//		}
	//	}
	//}




	//imwrite("optical_flow.png", img);
}

//
//#define ROW 480
//#define COL 640
//using namespace cv;
//using namespace std;
//void main() {
//
//
//	float cameraMatrix[3][3] = {
//		{ 3.367e+02,0,2.894e+02},
//		{0,3.352e+02,2.150e+02},
//		{0,0,1}
//	};//{fx 0 cx; 0 fy cy; 0 0 1}
//	float distCoeffs[5] = { 0.0589,-0.056, 0.0015,0.0023,0 };
//	//k1,k2,p1,p2,k3
//
//
//
////float cameraMatrix[3][3] = {
//	//	{ 3.367337955547257e+02,0,2.894704354849488e+02},
//	//	{0,3.352935387831358e+02,2.150233145155818e+02},
//	//	{0,0,1}
//	//};//{fx 0 cx; 0 fy cy; 0 0 1}
//	//float distCoeffs[5] = { 0.058941858532814,-0.056553014012356, 0.001526551383899,0.002388493872517,0 };
//	////k1,k2,p1,p2,k3
//	float fx, cx, fy, cy, k1, k2, p1, p2, k3;
//	fx = cameraMatrix[0][0];
//	cx = cameraMatrix[0][2];
//	fy = cameraMatrix[1][1];
//	cy = cameraMatrix[1][2];
//	k1 = distCoeffs[0];
//	k2 = distCoeffs[1];
//	p1 = distCoeffs[2];
//	p2 = distCoeffs[3];
//	k3 = distCoeffs[4];
//
//	Mat img1 = imread("nocal.jpg");
//	Mat img2 = imread("bb1.jpg");
//
//
//	Mat coordinateNorm(ROW,COL, CV_32FC2);
//	Mat coordinateDist(ROW, COL, CV_32FC2);
//	Mat coordinateMap(ROW, COL, CV_32FC2);
//	//1x,2y
//
//
//	//y=row竖线;x=col横线
//	for (int x = 0; x < COL-1; x++) {
//		for (int y = 0; y < ROW-1; y++) {
//			coordinateNorm.at<Vec2f>(y,x)[0] = (x - cx) / fx;
//			coordinateNorm.at<Vec2f>(y, x)[1] = (y - cy) / fy;
//		}
//	}
//	float xn, yn,r,r2,r4;
//	for (int x = 0; x < COL; x++) {
//		for (int y = 0; y < ROW; y++) {
//			xn = coordinateNorm.at<Vec2f>(y, x)[0];
//			yn = coordinateNorm.at<Vec2f>(y, x)[1];
//			r = sqrt(xn * xn + yn * yn);
//			r2 = pow(r, 2);
//			r4 = pow(r, 4);
//			coordinateDist.at<Vec2f>(y, x)[0] = xn * (1 + k1 * r2 + k2 * r4) + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
//			coordinateDist.at<Vec2f>(y, x)[1] = yn * (1 + k1 * r2 + k2 * r4) + 2 * p2 * xn * yn + p1 * (r2 + 2 * yn * yn);
//
//		}
//	}
//	float xd, yd;
//	float xmax = 0;
//	for (int x = 0; x < COL; x++) {
//		for (int y = 0; y < ROW; y++) {
//			xd = coordinateDist.at<Vec2f>(y, x)[0];
//			yd = coordinateDist.at<Vec2f>(y, x)[1];
//			coordinateMap.at<Vec2f>(y, x)[0] = fx * xd + cx;
//			coordinateMap.at<Vec2f>(y, x)[1] = fy * yd + cy;
//			
//			if (coordinateMap.at<Vec2f>(y, x)[0] > xmax && coordinateMap.at<Vec2f>(y, x)[0]<1000) {
//				xmax = coordinateMap.at<Vec2f>(y, x)[0];
//			}
//			//xmax is the border of x-axis
//
//			//cout << x << endl;
//			//cout << y << endl;
//			//cout << coordinateMap.at<Vec2f>(y, x)[0] << endl;
//			//cout << coordinateMap.at<Vec2f>(y, x)[1] << endl;
//
//		}
//	}
//	cout << xmax << endl;
//	Mat imgOutCal;
//	vector<Mat> channelsCal;
//	split(coordinateMap, channelsCal);
//	remap(img1, imgOutCal, channelsCal.at(0), channelsCal.at(1), INTER_LINEAR);
//	//================================distCal part====================================================
//	vector<Point2f> pts_src;
//	pts_src.push_back(Point2f(227, 182));
//	pts_src.push_back(Point2f(299, 180));
//	pts_src.push_back(Point2f(300, 235));
//	pts_src.push_back(Point2f(229, 238));
//	
//	vector<Point2f> pts_dst;
//	pts_dst.push_back(Point2f(280, 173));
//	pts_dst.push_back(Point2f(351, 171));
//	pts_dst.push_back(Point2f(352, 226));
//	pts_dst.push_back(Point2f(281, 229));
//
//	Mat h = findHomography(pts_src, pts_dst);
//	cout << h << endl;
//	/*float affineMatrix[3][3] = {
//	{ 0.934304301631126,-0.082751427110437,69.910362848197110},
//	{-0.003205934175658,0.899341263751184,1.976371226961151},
//	{0,0,1} 
//	};
//	float a, b, c, d, e, f;
//	a = affineMatrix[0][0];
//	b = affineMatrix[0][1];
//	c = affineMatrix[0][2];
//	d = affineMatrix[1][0];
//	e = affineMatrix[1][1];
//	f = affineMatrix[1][2];
//	cout << b << endl;*/
//	//Mat coordinateAffine(ROW, COL, CV_32FC2);
//
//	//float x1, y1;
//	//for (int x = 0; x < COL; x++) {
//	//	for (int y = 0; y < ROW; y++) {
//	//		x1 = coordinateMap.at<Vec2f>(y, x)[0];
//	//		y1 = coordinateMap.at<Vec2f>(y, x)[1];
//	//		coordinateAffine.at<Vec2f>(y, x)[0] = a*x+b*y+c;
//	//		coordinateAffine.at<Vec2f>(y, x)[1] = d*x+e*y+f;
//	//	}
//	//}
//	////[a b c; d e f; 0 0 1],[a b;c d]rotation, [c;f]translation
//	//Mat imgOutAffine;
//	//vector<Mat> channelsAffine;
//	//split(coordinateAffine, channelsAffine);
//	//remap(img2, imgOutAffine, channelsAffine.at(0), channelsAffine.at(1), INTER_LINEAR);
//	Mat imgOutAffine;
//
//	warpPerspective(img1, imgOutAffine, h, img2.size());
//
//
//	imwrite("cal.jpg", imgOutCal);
//
//
//
//
//	//imwrite("2.jpg", imgOutAffine);
//	imshow("1",imgOutCal);
//	imshow("2", imgOutAffine);
//
//
//	waitKey(0);
//
//
//
//	ofstream Fs("C:\\Users\\yrhxm\\Desktop\\test.xls");
//	if (!Fs.is_open())
//	{
//		cout << "error!" << endl;
//		return;
//	}
//
//
//	int height = coordinateMap.rows;
//	int width = coordinateMap.cols;
//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//		{
//			Fs << (int)coordinateMap.at<Vec2f>(i,j)[1] << '\t';
//		}
//		Fs << endl;
//	}
//	Fs.close();
//
//
//}