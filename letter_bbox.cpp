#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/edgepreserving_filter.hpp>
#include <filesystem>

using namespace std;
using namespace cv;

void fftshift(const Mat& inputImg, Mat& outputImg);
double GaussianCoeff(double u, double v, double d0);
double pixelDistance(double u, double v);
void performSUACE(Mat& src, Mat& dst, int distance, double sigma);
bool contourTouchesImageBorder(std::vector<cv::Point>& contour, cv::Size& imageSize);

Mat CreateGaussianFilter(cv::Mat& filter, Mat& complexI, int filter_type/* FilterType filter_type*/,
	float cutoff_f_low_InPixels, float cutoff_f_high_InPixels, float cutoff_f_center_InPixels);

int main( int argc, char** argv ) 
{	
	Mat imageMatrix,imageMatrix_8bit, complexI, outputImg, NormOutIfft, complexbandPass;
    double maxVal, minVal;
    Point max_loc, min_loc;
	int a = 5; 
	int b = 20; //b inceases contrast improves around features
	String save_path = "/home/jyothir-dr/Downloads/l2hackathonadasvideofunctioncomputervisiondevelope/output";
	String folderpath = "/home/jyothir-dr/Downloads/l2hackathonadasvideofunctioncomputervisiondevelope/testing/*.png" ;
	//images folder 

	vector<String> filenames;
	cv::glob(folderpath, filenames);

	for (size_t i=0; i<filenames.size(); i++)

	{
		Mat imageMatrix = imread(filenames[i]);
		string fileName = filenames[i];
		cout<<"fileName: "<<fileName<<endl;
		size_t position0 = fileName.find_last_of('/');
		string baseName0 = (string::npos == position0+1)? fileName : fileName.substr(position0+1, -1);
		size_t position = baseName0.find(".");
		string baseName = (string::npos == position)? baseName0 : baseName0.substr(0, position);
		cout<<"baseName: "<<baseName<<endl;
		string out_name = save_path+ "/" + baseName +"_res.png";
		if (imageMatrix.empty()) 
			{
				cout << "Image Not Found" << endl;
				// wait for any key press
				cin.get();
				return -1;
			}
		// cv::imshow("imageMatrix", imageMatrix);
		imageMatrix = cv::imread(fileName, IMREAD_GRAYSCALE);
		minMaxLoc(imageMatrix, &minVal, &maxVal, &min_loc, &max_loc);
		// cout<<"\n maxVal = "<<maxVal<<" minVal = "<<minVal;

		normalize(imageMatrix, imageMatrix_8bit, 0, 255, NORM_MINMAX, CV_8UC1);
		Scalar b_mean, b_stdDev;
		meanStdDev(imageMatrix_8bit, b_mean, b_stdDev);
		// cout<<"\n b_mean = "<<b_mean[0]<<" b_stdDev = "<<b_stdDev[0];

		imageMatrix_8bit.convertTo(imageMatrix_8bit, CV_32F);
		Rect roi = Rect(0, 0, imageMatrix_8bit.cols & -2, imageMatrix_8bit.rows & -2);
		imageMatrix_8bit = imageMatrix_8bit(roi);
		Mat planes[2] = { Mat_<float>(imageMatrix_8bit.clone()), Mat::zeros(imageMatrix_8bit.size(), CV_32F) };

		merge(planes, 2, complexI);
		dft(complexI, complexI);
		split(complexI, planes);
		planes[0].at<float>(0, 0) = 0;
		planes[1].at<float>(0, 0) = 0;
		outputImg = planes[0];

		// Create Gaussian mask for filtering frequencies
		Mat mag_out, gaussianKernal;
		gaussianKernal = CreateGaussianFilter(outputImg, mag_out, 0, 10, 50, 0); // mode, min, max

		fftshift(outputImg, outputImg);
		Mat planesbandPass[2] = { Mat_<float>(outputImg.clone()), Mat::zeros(outputImg.size(), CV_32F) };
		merge(planesbandPass, 2, complexbandPass);

		Mat complexIH;
		mulSpectrums(complexI, complexbandPass, complexIH, 0);
		complexbandPass.release();
		complexI.release();
		idft(complexIH, complexIH);
		split(complexIH, planes);
		complexIH.release();
		outputImg = planes[0];
		planes->release();
		normalize(outputImg, NormOutIfft, 0, 255, NORM_MINMAX, CV_8UC1);
		outputImg.release();

		// imshow("NormOutIfft", NormOutIfft);
		// good result for contour
		// Mat thresh_fft;
		// threshold(NormOutIfft, thresh_fft, 140, 255.0, ADAPTIVE_THRESH_MEAN_C); // good image
		// imshow("thresh_fft", thresh_fft);
		// imwrite("fftimage.png", NormOutIfft);

		a = 80; 
		b = 64;

		// psuace
		Mat clean_img_a, clean_img;
		performSUACE(NormOutIfft, clean_img_a, a, (b + 1) / 8.0);
		NormOutIfft.release();

		threshold(clean_img_a, clean_img, 127, 255.0, ADAPTIVE_THRESH_MEAN_C ); 
		clean_img_a.release();

		// string fname = baseName + to_string(a) + "_" + to_string(b)+".png";
		// string fthr = baseName + "thr.png";
		// cout<<"fname: "<<fname<<endl;
		// imshow("inpu", clean_img);
		// imwrite(fname, clean_img);

		Mat maskImg;
		clean_img.copyTo(maskImg);
		vector<Vec4i> hierarchy;
		Mat drawing;
		std::vector<std::vector<cv::Point> > contoursExt;
		cv::cvtColor(imageMatrix, drawing, cv::COLOR_GRAY2BGR);
		findContours(maskImg, contoursExt, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

		//cout << "\n contours:";
		cv::Size imageSize = maskImg.size();

		// bool liesOnBorder = contourTouchesImageBorder(contoursExt.at(0), imageSize);
		// // std::cout << "lies on border: " << std::to_string(liesOnBorder);
		// std::cout << " lies on border: " << liesOnBorder;
		// std::cout << std::endl;
		// // Draw contours
		// cv::drawContours(drawing, contoursExt, -1, cv::Scalar(255, 0, 255), 1);
		// // kernel11 = Mat::ones(51, 51, CV_8U);
		// int idx = 0;
		// for (auto c : contoursExt)
		// {
		// 	// cout<<"c:"<<c<<endl;
		// 	if (contourTouchesImageBorder(c, imageSize) == 1)
		// 		{
		// 			// cout << "inside loop"<<endl;
		// 			cv::drawContours(maskImg, contoursExt, idx, cv::Scalar(0, 0, 0), cv::FILLED);
		// 		}
		// 	idx++;
		// }

		for (auto c : contoursExt)
		{
			cv::Rect rect = cv::boundingRect(c);
			float ratio_hw = float(rect.height) / float(rect.width);
			if (rect.width<=20 && rect.width>4 && rect.height < imageSize.height-10 && ratio_hw > 1 )
			{
				rectangle(drawing, rect, Scalar(0, 255, 0), 1);
			// cout<<"height:" <<rect.height<<" width: "<< rect.width<<endl;
			}
		}
		// imshow("maskImg", maskImg);
		// imwrite(fthr, maskImg);
		// cv::waitKey();
		
		imwrite(out_name, drawing);
		clean_img.release(), drawing.release(), maskImg.release();

	}

	waitKey();
	destroyAllWindows();
}

void performSUACE(Mat& src, Mat& dst, int distance, double sigma)
{
	CV_Assert(src.type() == CV_8UC1);
	dst = Mat(src.size(), CV_8UC1);
	Mat smoothed;
	int val;
	int a, b;
	int adjuster;
	int half_distance = distance / 2;
	double distance_d = distance;

	GaussianBlur(src, smoothed, cv::Size(0, 0), sigma*2);
	//sigma*2 the more the image is smoothen, the more enhancement in contrast (features enhanced)
	//imshow("smoothed_image", smoothed);

	for (int x = 0; x < src.cols; x++)
		for (int y = 0; y < src.rows; y++) 
		{
			val = src.at<uchar>(y, x);
			adjuster = smoothed.at<uchar>(y, x);
			if ((val - adjuster) > distance_d)
				adjuster += (val - adjuster) * 0.5;
			adjuster = adjuster < half_distance ? half_distance : adjuster;
			b = adjuster + half_distance;
			b = b > 255 ? 255 : b;
			a = b - distance;
			a = a < 0 ? 0 : a;

			if (val >= a && val <= b)
			{
				dst.at<uchar>(y, x) = (int)(((val - a) / distance_d) * 255);
			}
			else if (val < a) {
				dst.at<uchar>(y, x) = 0;
			}
			else if (val > b) {
				dst.at<uchar>(y, x) = 255;
			}
		}
}

void fftshift(const Mat& inputImgfft, Mat& outputImgfft)
{
	Mat tmp;

	outputImgfft = inputImgfft.clone();

	int cx = outputImgfft.cols / 2;
	int cy = outputImgfft.rows / 2;

	Mat q0(outputImgfft, Rect(0, 0, cx, cy));
	Mat q1(outputImgfft, Rect(cx, 0, cx, cy));
	Mat q2(outputImgfft, Rect(0, cy, cx, cy));
	Mat q3(outputImgfft, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

double GaussianCoeff(double u, double v, double d0)
{
	double d = pixelDistance(u, v);
	return cv::exp((-d * d) / (2 * d0 * d0));
}

double pixelDistance(double u, double v)
{
	return cv::sqrt(u * u + v * v);
}

Mat CreateGaussianFilter(cv::Mat& outputImg, Mat& complexImg, int filter_type,
	float cutoff_f_low_InPixels, float cutoff_f_high_InPixels, float cutoff_f_center_InPixels)

{
	cv::Mat gf(outputImg.size(), CV_32F);
	cv::Point centre(outputImg.cols / 2, outputImg.rows / 2);

	for (int u = 0; u < gf.rows; u++)
	{
		for (int v = 0; v < gf.cols; v++)
		{
			gf.at<float>(u, v) = GaussianCoeff(u - centre.y, v - centre.x, cutoff_f_high_InPixels) -
				GaussianCoeff(u - centre.y, v - centre.x, cutoff_f_low_InPixels);
		}
	}

	// Keep the center pixel
	Mat gf1;
	Mat planesbandPass[2] = { Mat_<float>(gf.clone()), Mat::zeros(gf.size(), CV_32F) };
	Mat complexBp(gf.size(), CV_32F);
	outputImg = gf;
	return gf;
	//Mat filteredImg = {filter, complexImg};
	//merge(filteredImg, 2, complexImg);
	//waitKey();

}

bool contourTouchesImageBorder(std::vector<cv::Point>& contour, cv::Size& imageSize)
{
	cv::Rect bb = cv::boundingRect(contour);
	// cout<<imageSize<<endl;
	bool retval = false;
	if (bb.height>1 && bb.width>1)
	// cout<<"imageSize.height"<<imageSize.height;
	{
		float ratio_hw;
		// cout<<"height:"<<bb.height<<" width:"<<bb.width<<endl;
		// cout<<"bbx.x: "<<bb.x<<" bbx.y: "<<bb.y<<endl;
		ratio_hw = float(bb.height)/float(bb.width);
		// if (ratio_hw < 1)
		{
		// cout<<"\n ##### h-w ratio< 1:"<<ratio_hw<<" h near "<<bb.y + bb.height<<" w near "<<bb.x + bb.width<<endl;
		}
		int xMin, xMax, yMin, yMax ; //, xMininn, xMaxinn, yMininn, yMaxinn;

		xMin = 3;
		yMin = 10;
		xMax = imageSize.width - 0;
		yMax = imageSize.height - 0;
		// cout<<"bbx end: "<<bb.x + bb.width << " xmax: "<< xMax << " hgt:"<<bb.y + bb.height <<" ymax: " << yMax<<endl;
		if ((bb.x <= xMin || bb.y <= yMin )) //|| (bb.y + bb.height) <= 30 || (bb.x + bb.width) <= 90)) //|| ratio_hw < 0.5)) //|| (bb.x + bb.width) <= xMax || (bb.y + bb.height) <= yMax) )

		{
			retval = true;
		}
	}
	// cout<<" \n retval:"<<retval<<endl;
	return retval;
}
