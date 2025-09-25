#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

void augment_hsv(cv::Mat& image, float hgain = 0.5, float sgain = 0.5, float vgain = 0.5);

std::tuple<cv::Mat, std::vector<float>, std::vector<float>>  letterbox(cv::Mat img,
	std::pair<int, int> new_shape = { 640, 640 },
	cv::Scalar color = cv::Scalar(114, 114, 114),
	bool auto_mode = true,
	bool scaleFill = false,
	bool scaleup = true,
	int stride = 32);

std::tuple<cv::Mat, torch::Tensor, std::vector<torch::Tensor>> 
random_perspective(cv::Mat img, 
    torch::Tensor targets,
	std::vector<torch::Tensor> segments,
	int degrees, 
    float translate, 
    float scale,
	int shear, 
    float perspective, 
    std::vector<int> border);    