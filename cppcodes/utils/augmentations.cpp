#include "augmentations.h"
#include "utils.h"
#include "general.h"
#include "ops.h"

void augment_hsv(cv::Mat& image, float hgain/* = 0.5*/, float sgain/* = 0.5*/, float vgain/* = 0.5*/)
{
	// 生成随机增益 [-1,1]*gain + 1
	float r[3] = {
		random_uniform(-1.0f, 1.0f) * hgain + 1.0f,
		random_uniform(-1.0f, 1.0f) * sgain + 1.0f,
		random_uniform(-1.0f, 1.0f) * vgain + 1.0f
	};

	// 转换到HSV空间
	cv::Mat hsv;
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

	// 分离通道
	std::vector<cv::Mat> channels;
	cv::split(hsv, channels);

	// 创建LUT
	cv::Mat lut_hue(1, 256, CV_8U);
	cv::Mat lut_sat(1, 256, CV_8U);
	cv::Mat lut_val(1, 256, CV_8U);

	for (int i = 0; i < 256; ++i) {
		lut_hue.at<uchar>(i) = static_cast<uchar>(i * r[0]) % 180;
		lut_sat.at<uchar>(i) = cv::saturate_cast<uchar>(i * r[1]);
		lut_val.at<uchar>(i) = cv::saturate_cast<uchar>(i * r[2]);
	}

	// 应用LUT
	cv::LUT(channels[0], lut_hue, channels[0]);
	cv::LUT(channels[1], lut_sat, channels[1]);
	cv::LUT(channels[2], lut_val, channels[2]);

	// 合并通道并转换回BGR
	cv::merge(channels, hsv);
	cv::cvtColor(hsv, image, cv::COLOR_HSV2BGR);
}

std::tuple<cv::Mat, std::vector<float>, std::vector<float>>  letterbox(cv::Mat img,
	std::tuple<int, int> new_shape/* = { 640, 640 }*/,
	cv::Scalar color/* = cv::Scalar(114, 114, 114)*/,
	bool auto_mode/* = true*/,
	bool scaleFill/* = false*/,
	bool scaleup/* = true*/,
	int stride/* = 32*/) 
{
	auto [dst_width, dst_height] = new_shape;

	cv::Mat img_out = cv::Mat(dst_width, dst_height, CV_8UC3, cv::Scalar(114,114,114));
	std::vector<float> ratio = {1.f, 1.f};  // ratio [w, h]
	std::vector<float> pad ={0.f, 0.f};

	// load_image时已经根据比例放缩到了单边与imgsz一致了，暂时不考虑填充的模式
	auto h = img.rows;
	auto w = img.cols;

	auto new_h = h;
	auto new_w = w;

	if(scaleFill)	// stretch，将原图记放缩到与输出图像宽高一致，无边缘填充
	{
		pad = {0.f, 0.f};
		ratio[0] = float(dst_width) / float(w);
		ratio[1] = float(dst_height) / float(h);
		new_w = dst_width;
		new_h = dst_height;
	}
	else
	{
		auto w_r = float(dst_width) / float(w);
		auto h_r = float(dst_height) / float(h);
		auto min_ratio = std::min(w_r, h_r);
		ratio[0] = min_ratio;
		ratio[1] = min_ratio;
		new_w = std::min(dst_width, int(float(w) * ratio[0]));
		new_h = std::min(dst_height, int(float(h) * ratio[1]));
		pad[0] = (dst_width - new_w) / 2;
		pad[1] = (dst_height - new_h) / 2;
	}
	
	cv::Mat new_img;
	cv::resize(img, new_img, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);

	cv::Mat roi = img_out(cv::Rect(int(pad[0]), int(pad[1]), new_w, new_h));
	new_img.copyTo(roi);

	return std::make_tuple(img_out, ratio, pad);
}
