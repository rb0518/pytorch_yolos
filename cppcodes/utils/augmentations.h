#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

void augment_hsv(cv::Mat& image, float hgain = 0.5, float sgain = 0.5, float vgain = 0.5);

std::tuple<cv::Mat, std::vector<float>, std::vector<float>>  letterbox(cv::Mat img,
	std::tuple<int, int> new_shape = std::make_tuple( 640, 640 ),
	cv::Scalar color = cv::Scalar(114, 114, 114),
	bool auto_mode = true,
	bool scaleFill = false,
	bool scaleup = true,
	int stride = 32);
 