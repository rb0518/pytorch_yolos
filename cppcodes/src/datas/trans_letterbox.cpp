#include "trans_letterbox.h"

#include "utils.h"
#include "augmentations.h"

LetterBox::LetterBox(VariantConfigs* pargs, int imgsz, int dataset_num)
	: BaseTransform(pargs, imgsz, dataset_num)
{
}

void LetterBox::getSpecialSets()
{
}

void LetterBox::doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput)
{
	cv::Mat img = std::get<cv::Mat>(pInput.at(GET_VDK(VDK::IMAGE)));
	auto [w, h] = std::get < std::vector<std::tuple<float, float>>>(pInput.at(GET_VDK(VDK::SRC_SIZE)))[0];

	auto [img_out, ratio, pad] = letterbox(img, std::make_tuple(m_imageSize, m_imageSize));

	updateLabels(pInput, w, h, int(pad[0]), int(pad[1]));
	pInput["xyxy"] = int(1);
	pInput[GET_VDK(VDK::IMAGE)] = img_out;
	pInput[GET_VDK(VDK::RESIZED)] = std::vector<std::tuple<float, float>>({ std::make_tuple(float(img_out.cols), float(img_out.rows)) });
	pInput[GET_VDK(VDK::RATIO)] = changeTupleVector(ratio[0], ratio[1]);
	pInput[GET_VDK(VDK::PAD)] = changeTupleVector(pad[0], pad[1]);
}
#if 0
std::tuple<cv::Mat, std::vector<float>, std::vector<float>>  LetterBox::letterbox(cv::Mat img,
	cv::Scalar color/* = cv::Scalar(114, 114, 114)*/,
	bool auto_mode/* = true*/,
	bool scaleFill/* = false*/,
	bool scaleup/* = true*/,
	int stride/* = 32*/)
{
	cv::Mat img_out = cv::Mat(m_imageSize, m_imageSize, CV_8UC3, cv::Scalar(114, 114, 114));
	std::vector<float> ratio = { 1.f, 1.f };  // ratio [w, h]
	std::vector<float> pad = { 0.f, 0.f };

	// load_image时已经根据比例放缩到了单边与imgsz一致了，暂时不考虑填充的模式
	auto h = img.rows;
	auto w = img.cols;

	auto new_h = h;
	auto new_w = w;

	if (scaleFill)	// stretch，将原图记放缩到与输出图像宽高一致，无边缘填充
	{
		pad = { 0.f, 0.f };
		ratio[0] = float(m_imageSize) / float(w);
		ratio[1] = float(m_imageSize) / float(h);
		new_w = m_imageSize;
		new_h = m_imageSize;
	}
	else
	{
		auto w_r = float(m_imageSize) / float(w);
		auto h_r = float(m_imageSize) / float(h);
		auto min_ratio = std::min(w_r, h_r);
		ratio[0] = min_ratio;
		ratio[1] = min_ratio;
		new_w = std::min(m_imageSize, int(float(w) * ratio[0]));
		new_h = std::min(m_imageSize, int(float(h) * ratio[1]));
		pad[0] = (m_imageSize - new_w) / 2;
		pad[1] = (m_imageSize - new_h) / 2;
	}

	cv::Mat new_img;
	cv::resize(img, new_img, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);

	cv::Mat roi = img_out(cv::Rect(int(pad[0]), int(pad[1]), new_w, new_h));
	new_img.copyTo(roi);

	return std::make_tuple(img_out, ratio, pad);
}
#endif
