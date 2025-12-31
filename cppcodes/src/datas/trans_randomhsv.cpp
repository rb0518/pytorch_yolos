#include "trans_randomhsv.h"
#include "utils.h"
RandomHSV::RandomHSV(VariantConfigs* pargs, int imgsz, int dataset_num)
	: BaseTransform(pargs, imgsz, dataset_num)
{
}

void RandomHSV::getSpecialSets()
{
	m_hgain = getVariantValue(m_pArgs, "hsv_h", 0.5f);
	m_sgain = getVariantValue(m_pArgs, "hsv_s", 0.5f);
	m_vgain = getVariantValue(m_pArgs, "hsv_v", 0.5f);

	// 创建LUT
	float r[3] = {
		random_uniform(-1.0f, 1.f) * m_hgain,
		random_uniform(-1.0f, 1.f) * m_sgain + 1.f,
		random_uniform(-1.0f, 1.f) * m_vgain + 1.f
	};
	lut_hue = cv::Mat(1, 256, CV_8U);
	lut_sat = cv::Mat(1, 256, CV_8U);
	lut_val = cv::Mat(1, 256, CV_8U);

	for (int i = 0; i < 256; i++) {
		auto h = int(float(i) + r[0] * 180.f) % 180;
		h = std::min(255, std::max(0, h));

		auto s = int(float(i) * r[1]);
		s = std::min(255, std::max(0, s));

		auto v = int(float(i) * r[2]);
		v = std::min(255, std::max(0, v));


		lut_hue.at<uchar>(i) = static_cast<uchar>(h);
		lut_sat.at<uchar>(i) = static_cast<uchar>(s);
		lut_val.at<uchar>(i) = static_cast<uchar>(v);
	}
	lut_sat.at<uchar>(0) = 0;

}

void RandomHSV::doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput)
{
	cv::Mat img = std::get<cv::Mat>(pInput[GET_VDK(VDK::IMAGE)]);

	if (img.channels() != 3)
	{
		LOG(ERROR) << "RandomHSV only support RGB image";
		return;
	}


	cv::Mat hsv;
	cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> channels;
	cv::split(hsv, channels);

	// 应用LUT
	cv::LUT(channels[0], lut_hue, channels[0]);
	cv::LUT(channels[1], lut_sat, channels[1]);
	cv::LUT(channels[2], lut_val, channels[2]);

	// 合并通道并转换回BGR
	cv::merge(channels, hsv);
	cv::cvtColor(hsv, img, cv::COLOR_HSV2BGR);

	pInput[GET_VDK(VDK::IMAGE)] = img;
}
