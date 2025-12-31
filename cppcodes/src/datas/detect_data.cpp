#include "detect_data.h"
#include "utils.h"

#include "trans_mosaic.h"
#include "trans_letterbox.h"
#include "trans_randperspective.h"
#include "trans_randomhsv.h"
#include "trans_randomflip.h"
#include "trans_format.h"

void DetectTransformsArray::buildTranforms(bool bAugment)
{
	if (false == bAugment)
	{
		m_trans.push_back(std::dynamic_pointer_cast<BaseTransform>(std::make_shared<LetterBox>(m_pArgs, m_imageSize, m_datasetSize)));
		m_trans.push_back(std::dynamic_pointer_cast<BaseTransform>(std::make_shared<FormatOutput>(m_pArgs, m_imageSize, m_datasetSize)));
	}
	else
	{
		m_trans.push_back(std::dynamic_pointer_cast<BaseTransform>(std::make_shared<Mosaic>(
			m_pArgs, m_imageSize, m_datasetSize)));
		m_trans.push_back(std::dynamic_pointer_cast<BaseTransform>(std::make_shared<RandomPerspective>(
			m_pArgs, m_imageSize, m_datasetSize)));

		// 未像python代码添加的有 Mixup, CutMix, Albumentations
		m_trans.push_back(std::dynamic_pointer_cast<BaseTransform>(std::make_shared<RandomHSV>(
			m_pArgs, m_imageSize, m_datasetSize)));
		m_trans.push_back(std::dynamic_pointer_cast<BaseTransform>(std::make_shared<RandomFlip>(
			m_pArgs, m_imageSize, m_datasetSize)));

		m_trans.push_back(std::dynamic_pointer_cast<BaseTransform>(std::make_shared<FormatOutput>(
			m_pArgs, m_imageSize, m_datasetSize)));
	}

	for(auto& item : m_trans)
	{
		item->initialize();
	}
}


// ---------------------------------- DetectDataset Start -----------------------------------------

DetectDataset::DetectDataset(const std::string& path, VariantConfigs* pargs, int stride)
	: BaseDataset(path, pargs, stride)
{
	initFinal();
}

VariantData DetectDataset::get(size_t index)
{
	return getOneSample(index);
}

void DetectDataset::getSpecialSettins()
{
	m_pTransforms_Augment = std::make_shared<DetectTransformsArray>(
			m_pArgs, m_imgSize, m_imgFiles.size(), true);
	m_pTransforms_Normal = std::make_shared<DetectTransformsArray>(
			m_pArgs, m_imgSize, m_imgFiles.size(), false);

	m_pTransforms_Augment->buildTranforms(true);
	m_pTransforms_Normal->buildTranforms(false);
}

VariantData DetectDataset::getOneSample(size_t index)
{
	BaseDataset *pdataset = (BaseDataset *)this;
	auto data = getImageAndLabel(index);
	if (m_bAugment && m_bMosaic && getVariantValue(m_pArgs, "mosaic", 0.0f) == 1.0f)
	{
		m_pTransforms_Augment->doTransforms(pdataset, index, data);
	}
	else
	{
		m_pTransforms_Normal->doTransforms(pdataset, index, data);
	}
	return data;
}
// ---------------------------------- DetectDataset End -----------------------------------------


// ---------------------------------- DetectDataload Start -----------------------------------------

DetectDataload::DetectDataload(const std::string& root_path, VariantConfigs& args_input, int stride, bool is_val)
	: BaseDataloader(root_path, args_input, stride, is_val)	
{
}

void DetectDataload::createDataloader()
{
	auto datasets = DetectDataset(m_datasetsPath, &m_args, m_stride).map(DetectCustomCollate());
	m_totalNumber = datasets.size().value_or(0);
	if (m_totalNumber == 0)
		LOG(ERROR) << "datasets size == 0, check you dataset.yaml or datasets file.";

	int worker = getVariantValue<int>(&m_args, "workers", 1);
	auto dataloader_options = torch::data::DataLoaderOptions().batch_size(m_batchSize).workers(worker);
	m_dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(datasets), dataloader_options);
}

// ---------------------------------- DetectDataload End -----------------------------------------
