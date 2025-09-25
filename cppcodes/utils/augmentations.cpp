#include "augmentations.h"
#include "utils.h"
#include "general.h"

inline torch::Tensor box_candidates(const torch::Tensor& box1,
	const torch::Tensor& box2,
	float wh_thr = 2.0f,
	float ar_thr = 100.0f,
	float area_thr = 0.1f,
	float eps = 1e-16f) 
{
	auto w1 = box1[2] - box1[0];
	auto h1 = box1[3] - box1[1];
	auto w2 = box2[2] - box2[0];
	auto h2 = box2[3] - box2[1];

	auto ar1 = w2 / (h2 + eps);
	auto ar2 = h2 / (w2 + eps);
	auto ar = torch::maximum(ar1, ar2);

	auto area_ratio = (w2 * h2) / (w1 * h1 + eps);

	auto mask = (w2 > wh_thr) & (h2 > wh_thr) & (area_ratio > area_thr) & (ar < ar_thr);
	return mask;
}

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
	std::pair<int, int> new_shape/* = { 640, 640 }*/,
	cv::Scalar color/* = cv::Scalar(114, 114, 114)*/,
	bool auto_mode/* = true*/,
	bool scaleFill/* = false*/,
	bool scaleup/* = true*/,
	int stride/* = 32*/) 
{
	cv::Mat img_out = cv::Mat(new_shape.first, new_shape.second, CV_8UC3, cv::Scalar(114,114,114));
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
		ratio[0] = float(new_shape.first) / float(w);
		ratio[1] = float(new_shape.first) / float(h);
		new_w = new_shape.first;
		new_h = new_shape.first;
	}
	else
	{
		auto w_r = float(new_shape.first) / float(w);
		auto h_r = float(new_shape.first) / float(h);
		auto min_ratio = std::min(w_r, h_r);
		ratio[0] = min_ratio;
		ratio[1] = min_ratio;
		new_w = std::min(new_shape.first, int(float(w) * ratio[0]));
		new_h = std::min(new_shape.first, int(float(h) * ratio[1]));
		pad[0] = (new_shape.first - new_w) / 2;
		pad[1] = (new_shape.first - new_h) / 2;
	}
	
	cv::Mat new_img;
	cv::resize(img, new_img, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);

	cv::Mat roi = img_out(cv::Rect(int(pad[0]), int(pad[1]), new_w, new_h));
	new_img.copyTo(roi);

	return std::make_tuple(img_out, ratio, pad);
}

std::tuple<cv::Mat, torch::Tensor, std::vector<torch::Tensor>> 
random_perspective(cv::Mat img, 
    torch::Tensor targets,
	std::vector<torch::Tensor> segments,
	int degrees, 
    float translate, 
    float scale,
	int shear, 
    float perspective, 
    std::vector<int> border)
{
	// targets = [cls, xyxy]
	auto height = img.rows + border[0] * 2;
	auto width = img.cols + border[1] * 2;
	//std::cout << "           height width: [" << height << " " << width << "]" << std::endl;

	// Center
	auto C = torch::eye(3);
	C[0][2] = -(img.cols / 2);	// x translation(pixels)
	C[1][2] = -(img.rows / 2);	// y translation(pixels)
	//std::cout << "C: " << C << std::endl;

	auto P = torch::eye(3);
	P[2][0] = random_uniform(-perspective, perspective); // x perspective (about y)
	P[2][1] = random_uniform(-perspective, perspective); // y perspective (about x)
	//std::cout << "P: " << P << std::endl;

	// Rotation and Scale
	auto R = torch::eye(3);
	auto a = random_uniform(-degrees, degrees);
	auto s = random_uniform(1 - scale, 1 + scale);
	float rad = a * M_PI / 180.0f;
	float cos_val = s * cos(rad);
	float sin_val = s * sin(rad);

	// ������?����ǰ����
	R.index_put_({ 0, 0 }, cos_val);
	R.index_put_({ 0, 1 }, -sin_val);
	R.index_put_({ 1, 0 }, sin_val);
	R.index_put_({ 1, 1 }, cos_val);
	//std::cout << "R: " << R << std::endl;

	// Shear
	auto S = torch::eye(3);
	S[0][1] = tan(random_uniform(-shear, shear) * M_PI / 180); // x shear (deg)
	S[1][0] = tan(random_uniform(-shear, shear) * M_PI / 180); // y shear(deg)
	//std::cout << "S: " << S << std::endl;

	// Translation
	auto T = torch::eye(3);
	T[0][2] = random_uniform(0.5 - translate, 0.5 + translate) * width;  // x translation(pixels)
	T[1][2] = random_uniform(0.5 - translate, 0.5 + translate) * height;  // y translation(pixels)
	//std::cout << "T: " << T << std::endl;

	// 注意，这段代码顺序不能乱
	auto M = T.matmul(S).matmul(R).matmul(P).matmul(C);

	bool need_transform = border[0]!= 0 || border[1]!=0 ||
		(M != torch::eye(3, torch::kFloat32)).any().item<bool>();
	//std::cout << "need_transform : " << need_transform << std::endl;
	//std::cout << "M : " << M << std::endl;

	if (need_transform)
	{
		cv::Mat cv_M(3, 3, CV_32F);
		cv::Scalar border_value = { 114, 114, 114 };
		std::memcpy(cv_M.data, M.data_ptr(), M.numel() * sizeof(float));

		cv::Mat output;
		if (perspective) {
			cv::warpPerspective(img, output, cv_M, cv::Size(width, height),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
		}
		else {
			cv::Mat affine_M = cv_M(cv::Rect(0, 0, 3, 2));
			cv::warpAffine(img, output, affine_M, cv::Size(width, height),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
		}
		img = output;
	}
	//std::cout << "after rand_perspective: [" << img.cols << " " << img.rows << "]" << std::endl;
	// Transform label coordinates
	std::vector<torch::Tensor> new_segments;
	std::vector<torch::Tensor> result_seg;
	auto result_tag = targets.clone();
	int n = targets.size(0);
	bool use_segments = true;
    if(segments.size() == 0 || segments.size() != n)
        use_segments = false;
	if (n)
	{
		auto newbox = torch::zeros({ n, 4 });
		if (use_segments)
		{
			auto segments_r = resample_segments(segments);

			for (int i = 0; i < segments_r.size(); i++)
			{
				auto segment = segments_r[i];	// [2, 1000]
				auto xy = torch::ones({ segment.size(0), 3 });
				xy.index_put_({ "...", torch::indexing::Slice(0, 2) }, segment);
				xy = xy.matmul(M.t());		//  xy = xy @ M.T # transoform

				if (perspective)
				{
					xy.index_put_({ "...", torch::indexing::Slice(0, 2) },
						xy.index({ "...", torch::indexing::Slice(0, 2) }) / xy.index({ "...", torch::indexing::Slice(2, 3) }));
				}
				auto xy_2 = xy.index({ "...", torch::indexing::Slice(0, 2) });
				auto segbox = segment2box(xy_2, width, height);
				newbox.index_put_({ i, torch::indexing::Slice() }, segbox);
				new_segments.push_back(xy_2);
			}
		}
		else
		{
			auto xy = torch::ones({ n * 4, 3 }, torch::kFloat32);

			// 重构数据组织 (x1y1, x2y2, x1y2, x2y1)
			auto target_coords = targets.index({ torch::indexing::Slice(),
											  torch::tensor({1, 2, 3, 4, 1, 4, 3, 2}) });	// {x1y1, x2y2, x1y2, x2y1}
			xy.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0, 2) },
				target_coords.reshape({ n * 4, 2 }));

			// 坐标体系与图像同等transform 
			xy = xy.matmul(M.t());

			// 是否透视效果
			if (perspective) {
				auto xy_div = xy.index({ torch::indexing::Slice(), 2 }).unsqueeze(1);
				xy = xy.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 2) }).div(xy_div);
			}
			else {
				xy = xy.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 2) });
			}
			xy = xy.reshape({ n, 8 });
			// create new boxes
			auto x = xy.index({ torch::indexing::Slice(),
							  torch::tensor({0, 2, 4, 6}) });
			auto y = xy.index({ torch::indexing::Slice(),
							  torch::tensor({1, 3, 5, 7}) });

			auto x_min = std::get<0>(x.min(1));
			auto y_min = std::get<0>(y.min(1));
			auto x_max = std::get<0>(x.max(1));
			auto y_max = std::get<0>(y.max(1));

			newbox = torch::stack({ x_min, y_min, x_max, y_max }, 1);
		}

		
		auto box1 = targets.index({ torch::indexing::Slice(),
						 torch::indexing::Slice(1, 5) }).t() * s;
		auto box2 = newbox.t();

		//std::cout << "box1: " << box1.sizes() << " box2: " << box2.sizes() << std::endl;
		float thr = use_segments ? 0.01f : 0.10f;
		auto keep = box_candidates(box1, box2, thr);
		result_tag = targets.index({ keep });
		result_tag.index_put_({ torch::indexing::Slice(),
						  torch::indexing::Slice(1, 5) },
			newbox.index({ keep }));
		
		if (use_segments)
		{
			for (int i = 0; i < keep.size(0); i++)
			{
				if (keep[i].item<bool>())
					result_seg.push_back(new_segments[i]);
			}
		}
		else
		{	// 这一步是可以不做的，不是segments数据源，后续也不会使用
			for (int i = 0; i < keep.size(0); i++)
			{
				if (keep[i].item<bool>() && segments.size() == n)
					result_seg.push_back(segments[i]);
			}
		}
	}
	else
	{
		return { img, targets.clone(), segments};
	}

	return { img, result_tag, result_seg }; //std::make_tuple(img, targets, segments);	
}
