#include "trans_format.h"
#include "ops.h"
#include "utils.h"
FormatOutput::FormatOutput(VariantConfigs* pargs,
	int imgsz, int dataset_num)
	: BaseTransform(pargs, imgsz, dataset_num)
{
}

void FormatOutput::getSpecialSets()
{
	m_overlayMask = getVariantValue<bool>(m_pArgs, "overlap_mask", true);
	m_maskRatio = float(getVariantValue<int>(m_pArgs, "mask_ratio", 4));
}

void FormatOutput::doTransform(BaseDataset* pdataset, size_t index, VariantData& pInput)
{
	cv::Mat image = std::get<cv::Mat>(pInput[GET_VDK(VDK::IMAGE)]);
	torch::Tensor cls = std::get<torch::Tensor>(pInput[GET_VDK(VDK::CLS)]);
	torch::Tensor bboxes = std::get<torch::Tensor>(pInput[GET_VDK(VDK::BBOXES)]);
	bool bxyxy = std::get<int>(pInput["xyxy"]) > 0;

	torch::Tensor segments;
	bool bHaveSegment = pInput.count("segments");
	if (bHaveSegment)
		segments = std::get<torch::Tensor>(pInput["segments"]);

	int w = image.cols;
	int h = image.rows;

	int nl = bboxes.size(0);

	if (bxyxy)
		bboxes = xyxyn2xywh(bboxes, w, h);
	else
		LOG(ERROR) << "program run to here, the bboxes must be xyxy and denorm. ";

	auto img = convertImageToTensor(image);
	pInput[GET_VDK(VDK::IMG)] = img;

	if (bHaveSegment)
	{
		torch::Tensor masks, sorted_idx;
		if (nl == 0)
		{
			int ndim_0 = m_overlayMask ? 1 : nl;
			torch::ScalarType to = m_overlayMask ? torch::kInt16 : torch::kByte;
			masks = torch::zeros({ ndim_0, image.rows/int(m_maskRatio), image.cols/int(m_maskRatio) }, to);
		}
		else
		{
			if (m_overlayMask)
			{
				std::tie(masks, sorted_idx) = polygons2masks_overlap(cv::Size(w, h), segments, m_maskRatio);
				masks = masks.unsqueeze(0);	// [640,640] ==> [1, 640, 640]
				bboxes = bboxes.index_select(0, sorted_idx);
				cls = cls.index_select(0, sorted_idx);
			}
			else
			{
				masks = polygons2masks(cv::Size(w, h), segments, m_maskRatio);
			}
		}
		pInput[GET_VDK(VDK::MASK)] = masks;
	}

	pInput[GET_VDK(VDK::BBOXES)] = bboxes;
	pInput[GET_VDK(VDK::CLS)] = cls;
	pInput[GET_VDK(VDK::BATCH_IDX)] = torch::zeros({ bboxes.size(0) }, torch::kByte).view({ -1, 1 });
}

torch::Tensor FormatOutput::convertImageToTensor(cv::Mat image)
{
	int channels = image.channels();

	if(channels == 1 || channels == 3)	// 本程序不会出现==1, imread时都转换了
	{
		if (channels == 3)
			cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
		else
			cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
	}
	else
	{
		LOG(ERROR) << "image type error, channels: " << channels;
		return torch::zeros({ 3,m_imageSize, m_imageSize });
	}

	torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte);
	img_tensor = img_tensor.permute({ 2, 0, 1 });	// [h,w, c] ==> [c, h, w]
	return img_tensor.clone();
}
