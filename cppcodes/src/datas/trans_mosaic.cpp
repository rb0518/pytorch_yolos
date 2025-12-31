#include <algorithm>

#include "trans_mosaic.h"
#include "BaseData.h"

#include "utils.h"
#include "trans_letterbox.h"

// ------------------------------------ Mosaic Start ------------------
Mosaic::Mosaic(VariantConfigs* pargs,
	int imgsz,
	int dataset_num) : BaseTransform(pargs, imgsz, dataset_num)
{
}

void Mosaic::getSpecialSets()
{
    m_mosaicProb = getVariantValue<float>(this->m_pArgs, "mosaic", 1.0f);
    m_border = {- m_imageSize/2.f, - m_imageSize/2.f};
}

/*
	mosaic 返回坐标为不是在原始图上的坐标，转换为mosaic上的坐标，是xyxy格式
*/
void Mosaic::doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput)
{
	// 1 get images and labels;
	std::vector<int> indexes = getIndexes(index);

	std::vector<VariantData> datas;
	datas.push_back(pInput);

	for (int i = 1; i < indexes.size(); i++)
	{
		datas.push_back(pdataset->getImageAndLabel(indexes[i]));
	}

	int yc = int(random_uniform(-m_border[0], 2 * m_imageSize + m_border[0]));
	int xc = int(random_uniform(-m_border[0], 2 * m_imageSize + m_border[0]));
	cv::Mat img4 = cv::Mat(m_imageSize * 2, m_imageSize * 2, CV_8UC3, cv::Scalar(114, 114, 114));

	for (int i = 0; i < 4; i++)
	{
		cv::Mat image = std::get<cv::Mat>(datas[i][GET_VDK(VDK::IMAGE)]);
		int h = image.rows;
		int w = image.cols;

		int x1a, y1a, x2a, y2a;
		int x1b, y1b, x2b, y2b;

		if (i == 0)	// top left
		{
			x1a = std::max(xc - w, 0);		// 随机中心点左上偏移
			y1a = std::max(yc - h, 0);
			x2a = xc;
			y2a = yc;
			x1b = w - (x2a - x1a);
			y1b = h - (y2a - y1a);
			x2b = w;
			y2b = h;
		}
		else if (i == 1)// top right
		{
			x1a = xc;
			y1a = std::max(yc - h, 0);
			x2a = std::min(xc + w, m_imageSize * 2);
			y2a = yc;
			x1b = 0;
			y1b = h - (y2a - y1a);
			x2b = std::min(w, x2a - x1a);
			y2b = h;
		}
		else if (i == 2) {	// bottom left
			x1a = std::max(xc - w, 0);
			y1a = yc;
			x2a = xc;
			y2a = std::min(m_imageSize * 2, yc + h);
			x1b = w - (x2a - x1a);
			y1b = 0;
			x2b = w;
			y2b = std::min(y2a - y1a, h);
		}
		else if (i == 3) {
			x1a = xc;
			y1a = yc;
			x2a = std::min(xc + w, m_imageSize * 2);
			y2a = std::min(m_imageSize * 2, yc + h);
			x1b = 0;
			y1b = 0;
			x2b = std::min(w, x2a - x1a);
			y2b = std::min(y2a - y1a, h);
		}
		cv::Mat roi = img4(cv::Rect(x1a, y1a, x2a - x1a, y2a - y1a));
		image(cv::Rect(x1b, y1b, x2b - x1b, y2b - y1b)).copyTo(roi);

		int padw = x1a - x1b;
		int padh = y1a - y1b;

		updateLabels(datas[i], w, h, padw, padh);
	}

	// 合并信息
	// std::vector<std::string> keys;
	// std::transform(pInput.begin(), pInput.end(), std::back_inserter(keys), [](const auto& pair) {
	// 	return pair.first; });
	// std::for_each(keys.begin(), keys.end(), [](const auto& k) {std::cout << "key: " << k << std::endl; });


	std::vector<torch::Tensor> clses;
	std::vector<torch::Tensor> bboxes;
	std::vector<torch::Tensor> segments;
	bool have_segments = datas[0].count(GET_VDK(VDK::SEGMENTS)) > 0;
	for (int i = 0; i < 4; i++)
	{
		clses.push_back(std::get<torch::Tensor>(datas[i][GET_VDK(VDK::CLS)]).view({ -1, 1 }));
		bboxes.push_back(std::get<torch::Tensor>(datas[i][GET_VDK(VDK::BBOXES)]));
		if (have_segments)
		{
			torch::Tensor segment = std::get<torch::Tensor>(datas[i][GET_VDK(VDK::SEGMENTS)]);
			segments.emplace_back(segment);
		}
	}
	
	pInput[GET_VDK(VDK::BBOXES)] = torch::cat(bboxes, 0);
	pInput[GET_VDK(VDK::CLS)] = torch::cat(clses, 0);

	if (have_segments)
	{
		pInput[GET_VDK(VDK::SEGMENTS)] = torch::cat(segments, 0);
	}

	pInput["xyxy"] = 1;	// 坐标已经转换为xyxy了
	pInput[GET_VDK(VDK::IMAGE)] = img4;
	pInput[GET_VDK(VDK::RESIZED)] = changeTupleVector(img4.cols, img4.rows);

	// 添加mosaic处理变量
	pInput[GET_VDK(VDK::MOSAIC_BORDER)] = m_border;
}


std::vector<int> Mosaic::getIndexes(size_t index)
{
	std::vector<int> indexes;
	indexes.push_back(int(index));

	for (int i = 0; i < 3; i++)
		indexes.push_back(int(random_uniform(0, m_datasetSize - 1)));

	return indexes;
}
