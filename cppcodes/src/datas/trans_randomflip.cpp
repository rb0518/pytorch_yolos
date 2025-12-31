#include "trans_randomflip.h"
#include "utils.h"
RandomFlip::RandomFlip(VariantConfigs* pargs, int imgsz, int dataset_num)
	: BaseTransform(pargs, imgsz, dataset_num)
{
}

void RandomFlip::getSpecialSets()
{
	m_flipudProb = getVariantValue(m_pArgs, "flipud", 0.5f);
	m_fliplrProb = getVariantValue(m_pArgs, "fliplr", 0.5f);
}

void RandomFlip::doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput)
{
	cv::Mat img = std::get<cv::Mat>(pInput[GET_VDK(VDK::IMAGE)]);
	bool bxyxy = std::get<int>(pInput["xyxy"]) > 0;

	float width = float(img.cols);
	float height = float(img.rows);

	float h = height;
	float w = width;

	torch::Tensor bboxes = std::get<torch::Tensor>(pInput[GET_VDK(VDK::BBOXES)]);

	bool bHaveSegments = false;
	torch::Tensor segments;
	if (pInput.count("segments") > 0)
	{
		segments = std::get<torch::Tensor>(pInput["segments"]);
		bHaveSegments = true;
	}

	if (random_uniform() < m_flipudProb)
	{	// up_down 
		cv::flip(img, img, 0);
		if (bxyxy)
		{
			auto y1 = bboxes.index({ torch::indexing::Slice(), 1 }).clone();
			auto y2 = bboxes.index({ torch::indexing::Slice(), 3 }).clone();
			bboxes.index_put_({ torch::indexing::Slice(), 1 }, h - y2);
			bboxes.index_put_({ torch::indexing::Slice(), 3 }, h - y1);
		}
		else
		{
			bboxes.index_put_({ torch::indexing::Slice(), 1 }, h - bboxes.index(
				{torch::indexing::Slice(), 1}));
		}

		if (bHaveSegments)
			segments.index_put_({ "...", 1 }, h - segments.index({ "...", 1 }));
	}

	if (random_uniform() < m_fliplrProb) 
	{	// left_right
		cv::flip(img, img, 1);
		if (bxyxy)
		{
			auto x1 = bboxes.index({ torch::indexing::Slice(), 0 }).clone();
			auto x2 = bboxes.index({ torch::indexing::Slice(), 2 }).clone();
			bboxes.index_put_({ torch::indexing::Slice(), 0 }, w - x2);
			bboxes.index_put_({ torch::indexing::Slice(), 2 }, w - x1);
		}
		else
		{
			bboxes.index_put_({ torch::indexing::Slice(), 0 }, w - bboxes.index(
				{ torch::indexing::Slice(), 0 }));
		}

		if (bHaveSegments)
			segments.index_put_({ "...", 0 }, w - segments.index({ "...", 0 }));
	}
	pInput[GET_VDK(VDK::IMAGE)] = img;
	pInput[GET_VDK(VDK::BBOXES)] = bboxes;
	if (bHaveSegments)
		pInput["segments"] = segments;
}
