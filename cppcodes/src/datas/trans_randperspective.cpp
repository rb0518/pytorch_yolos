#include "trans_randperspective.h"
#include "utils.h"
#include "ops.h"

RandomPerspective::RandomPerspective(VariantConfigs* pargs, int imgsz, int dataset_num)
	: BaseTransform(pargs, imgsz, dataset_num)
{
	m_size.push_back(0.0f);
	m_size.push_back(0.0f);
}

void RandomPerspective::getSpecialSets()
{
	float m_degrees = getVariantValue<float>(m_pArgs, "degrees", 0.0f);
	float m_translate = getVariantValue<float>(m_pArgs, "translate", 0.0f);
	float m_scale = getVariantValue<float>(m_pArgs, "scale", 0.0f);
	float m_shear = getVariantValue<float>(m_pArgs, "shear", 0.0f);
	float m_perspective = getVariantValue<float>(m_pArgs, "perspective", 0.0f);
}

void RandomPerspective::doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput)
{
	std::vector<float> border = std::vector<float>({ 0.0f, 0.0f });
	if (pInput.count(GET_VDK(VDK::MOSAIC_BORDER)) > 0)
	{
		border = std::get<std::vector<float>>(pInput[GET_VDK(VDK::MOSAIC_BORDER)]);
	}

	// std::cout << "RandomPerspective: " << std::endl;
	// std::vector<std::string> keys;
	// std::transform(pInput.begin(), pInput.end(), std::back_inserter(keys), [](const auto& pair) {
	// 	return pair.first; });
	// std::for_each(keys.begin(), keys.end(), [](const auto& k) {std::cout << "key: " << k << std::endl; });

	cv::Mat image = std::get<cv::Mat>(pInput[GET_VDK(VDK::IMAGE)]);
	
	m_size[0] = float(image.cols) + border[0] * 2;	// width
	m_size[1] = float(image.rows) + border[1] * 2;	// height

	auto [img_out, M, scale] = affine_transform(image, border);
	pInput[GET_VDK(VDK::IMAGE)] = img_out;

	torch::Tensor bbox = apply_bboxes(pInput, M);
	torch::Tensor segments;
	if (pInput.count("segments") > 0)
	{
		std::tie(bbox, segments) = apply_segments(pInput, M);
	}

	// 再次裁切
	bbox.index_put_({ torch::indexing::Slice(), 0 }, bbox.index({ torch::indexing::Slice(), 0 }).clip(0, m_size[0]));
	bbox.index_put_({ torch::indexing::Slice(), 2 }, bbox.index({ torch::indexing::Slice(), 2 }).clip(0, m_size[0]));
	bbox.index_put_({ torch::indexing::Slice(), 1 }, bbox.index({ torch::indexing::Slice(), 1 }).clip(0, m_size[1]));
	bbox.index_put_({ torch::indexing::Slice(), 3 }, bbox.index({ torch::indexing::Slice(), 3 }).clip(0, m_size[1]));
	if (segments.size(0))
	{
		segments.index_put_({ "...", 0 }, bbox.index({ "...", 0 }).clip(0, m_size[0]));
		segments.index_put_({ "...", 1 }, bbox.index({ "...", 1 }).clip(0, m_size[1]));
	}

	// 同等比例准备
	torch::Tensor old_box = std::get<torch::Tensor>(pInput[GET_VDK(VDK::BBOXES)]);
	old_box.mul_(torch::tensor({ scale, scale, scale, scale }));


	float area_thr = 0.1f;
	if (pInput.count("segments") > 0)
		area_thr = 0.01f;

	auto index_mask = box_candidates(old_box.t(), bbox.t(), area_thr);

	pInput[GET_VDK(VDK::BBOXES)] = bbox.index({ index_mask });
	pInput[GET_VDK(VDK::CLS)] = std::get<torch::Tensor>(pInput[GET_VDK(VDK::CLS)]).index({ index_mask });
	if (pInput.count("segments") > 0)
		pInput["segments"] = segments.index({ index_mask });
}
#if 1		// 用cv::Mat::eye与python代码更接近
std::tuple<cv::Mat, torch::Tensor, float> RandomPerspective::affine_transform(cv::Mat img, std::vector<float> border)
{	
	float ic = float(img.cols);
	float ir = float(img.rows);
	// Center
	cv::Mat C = cv::Mat::eye(3, 3, CV_32F);
	C.at<float>(0, 2) = -(ic / 2.f);
	C.at<float>(1, 2) = -(ir / 2.f);

	// Perspective
	cv::Mat P = cv::Mat::eye(3, 3, CV_32F);
	P.at<float>(2, 0) = random_uniform(-m_perspective, m_perspective);
	P.at<float>(2, 1) = random_uniform(-m_perspective, m_perspective);

	// Rotation and Scale
	float a = random_uniform(-m_degrees, m_degrees);
	float s = random_uniform(1 - m_scale, 1 + m_scale);
	cv::Mat m_rm = cv::getRotationMatrix2D(cv::Point2f(0.f, 0.f), double(a), double(s));
	cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
	m_rm.row(0).copyTo(R.row(0));
	m_rm.row(1).copyTo(R.row(1));

	// Shear
	cv::Mat S = cv::Mat::eye(3, 3, CV_32F);
	S.at<float>(0, 1) = tan(random_uniform(-m_shear, m_shear) * M_PI / 180);
	S.at<float>(1, 0) = tan(random_uniform(-m_shear, m_shear) * M_PI / 180);
	
	// Translation
	cv::Mat T = cv::Mat::eye(3, 3, CV_32F);
	T.at<float>(0, 2) = random_uniform(0.5 - m_translate, 0.5 + m_translate) * m_size[0];
	T.at<float>(1, 2) = random_uniform(0.5 - m_translate, 0.5 + m_translate) * m_size[1];

	cv::Mat M = T*S*R*P*C;

	auto isIdentityMatrix = [](const cv::Mat& mat) ->bool {
		if (mat.type() != CV_32F) {
			throw std::invalid_argument("type error!");
		}
		cv::Mat eye = cv::Mat::eye(mat.rows, mat.cols, mat.type());
		return cv::countNonZero(mat != eye) == 0;
		};

	bool need_transform = border[0] != 0 || border[1] != 0 || isIdentityMatrix(M);
	if (need_transform)
	{
		cv::Scalar border_value = { 114, 114, 114 };

		cv::Mat output;
		if (m_perspective) {
			cv::warpPerspective(img, output, M, cv::Size(m_size[0], m_size[1]),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
		}
		else {
			cv::Mat affine_M = cv::Mat::zeros(2, 3, CV_32F);
			M.row(0).copyTo(affine_M.row(0));
			M.row(1).copyTo(affine_M.row(1));
			cv::warpAffine(img, output, affine_M, cv::Size(m_size[0], m_size[1]),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
		}
		img = output;
	}

	// cv::Mat => torch::Tensor for apply_bboxes() and apply_segments()
	torch::Tensor t_M = torch::from_blob(M.data, { M.rows, M.cols }, torch::kFloat32);
	return { img, t_M, s };
}
#else
std::tuple<cv::Mat, torch::Tensor, float> RandomPerspective::affine_transform(cv::Mat img, std::vector<float> border)
{
	// targets = [cls, xyxy]
	auto height = m_size[1];
	auto width = m_size[0];

	// Center
	auto C = torch::eye(3);
	C[0][2] = -(img.cols / 2);	// x translation(pixels)
	C[1][2] = -(img.rows / 2);	// y translation(pixels)
	//std::cout << "C: " << C << std::endl;

	auto P = torch::eye(3);
	P[2][0] = random_uniform(-m_perspective, m_perspective); // x perspective (about y)
	P[2][1] = random_uniform(-m_perspective, m_perspective); // y perspective (about x)
	//std::cout << "P: " << P << std::endl;

	// Rotation and Scale
	auto R = torch::eye(3);
	auto a = random_uniform(-m_degrees, m_degrees);
	auto s = random_uniform(1 - m_scale, 1 + m_scale);
	float rad = a * M_PI / 180.0f;
	float cos_val = s * cos(rad);
	float sin_val = s * sin(rad);
	R.index_put_({ 0, 0 }, cos_val);
	R.index_put_({ 0, 1 }, -sin_val);
	R.index_put_({ 1, 0 }, sin_val);
	R.index_put_({ 1, 1 }, cos_val);
	//std::cout << "R: " << R << std::endl;

	// Shear
	auto S = torch::eye(3);
	S[0][1] = tan(random_uniform(-m_shear, m_shear) * M_PI / 180); // x shear (deg)
	S[1][0] = tan(random_uniform(-m_shear, m_shear) * M_PI / 180); // y shear(deg)
	//std::cout << "S: " << S << std::endl;

	// Translation
	auto T = torch::eye(3);
	T[0][2] = random_uniform(0.5 - m_translate, 0.5 + m_translate) * width;  // x translation(pixels)
	T[1][2] = random_uniform(0.5 - m_translate, 0.5 + m_translate) * height;  // y translation(pixels)
	//std::cout << "T: " << T << std::endl;

	// 注意，这段代码顺序不能乱
	auto M = T.matmul(S).matmul(R).matmul(P).matmul(C);

	bool need_transform = border[0] != 0 || border[1] != 0 ||
		(M != torch::eye(3, torch::kFloat32)).any().item<bool>();
	//std::cout << "need_transform : " << need_transform << std::endl;
	//std::cout << "M : " << M << std::endl;

	cv::Mat output;
	if (need_transform)
	{
		cv::Mat cv_M(3, 3, CV_32F);
		cv::Scalar border_value = { 114, 114, 114 };
		std::memcpy(cv_M.data, M.data_ptr(), M.numel() * sizeof(float));


		if (m_perspective > 0.01f) {
			cv::warpPerspective(img, output, cv_M, cv::Size(width, height),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
		}
		else {
			cv::Mat affine_M = cv_M(cv::Rect(0, 0, 3, 2));
			cv::warpAffine(img, output, affine_M, cv::Size(width, height),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
		}
	}

	return { output, M, s };
}
#endif

torch::Tensor RandomPerspective::apply_bboxes(VariantData& labels, torch::Tensor M)
{
	torch::Tensor bbox = std::get<torch::Tensor>(labels[GET_VDK(VDK::BBOXES)]);
	int n = bbox.size(0);
	if (n == 0)
		return bbox;

	auto xy = torch::ones({ n * 4, 3 }, torch::kFloat32);
	// 重构数据组织 (x1y1, x2y2, x1y2, x2y1)
	auto target_coords = bbox.index({ torch::indexing::Slice(),
									torch::tensor({0, 1, 2, 3, 0, 3, 2, 1}) }).reshape({ n * 4, 2 });
	xy.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0, 2) },
		target_coords);

	// 坐标体系与图像同等transform 
	xy = xy.matmul(M.t());

	// 是否透视效果
	if (m_perspective > 0.0f) {
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

	auto new_boxes = torch::stack({ x_min, y_min, x_max, y_max }, 1);
	return new_boxes;
}

std::tuple<torch::Tensor, torch::Tensor> RandomPerspective::apply_segments(VariantData& labels, torch::Tensor M)
{
	torch::Tensor segments = std::get<torch::Tensor>(labels["segments"]);
	int n = segments.size(0);
	int num = segments.size(1);

	if (0 == n)
		return { torch::zeros({0, 4}, torch::kFloat32), segments };

	torch::Tensor xy = torch::ones({ n * num, 3 }, torch::kFloat32);
	segments = segments.reshape({ -1, 2 });
	xy.index_put_({ "...", torch::indexing::Slice(0, 2) }, segments);
	xy = xy.matmul(M.t());
	auto xy_div = xy.index({ "...", torch::indexing::Slice(2, 3) }).unsqueeze(1);
	auto xy_div_re = xy.index({ "...", torch::indexing::Slice(0, 2) }).div(xy_div);
	segments = xy_div_re.reshape({ n, -1, 2 });

	std::vector<torch::Tensor> boxes;
	for (int i = 0; i < n; i++) {
		boxes.push_back(segment2box(segments[i], m_size[0], m_size[1]));
	}
	auto bboxes = torch::stack(boxes, 0);

	segments.index_put_({ "...", 0 }, segments.index({ "...", 0 }).clip(
				bboxes.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)}),
				bboxes.index({ torch::indexing::Slice(), torch::indexing::Slice(2, 3)})
		));

	segments.index_put_({ "...", 1 }, segments.index({ "...", 1 }).clip(
				bboxes.index({ torch::indexing::Slice(), torch::indexing::Slice(1, 2) }),
				bboxes.index({ torch::indexing::Slice(), torch::indexing::Slice(3, 4) })
		));


	return { bboxes, segments };
}

torch::Tensor RandomPerspective::box_candidates(const torch::Tensor box1,
	const torch::Tensor box2,
	float area_thr /*= 0.1f*/,
	float wh_thr/* = 2.0f*/,
	float ar_thr/* = 100.0f*/,
	float eps /*= 1e-16f*/)
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
