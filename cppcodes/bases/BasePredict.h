#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "globaldef.h"
#include "BaseModel.h"
/*
*/
class BasePredict
{
public:
	explicit BasePredict(VariantConfigs& args_input, std::string root_dir);
	void setupPredict();
	void doPredictTask();

protected:
	void readSettings();
	void initDevice();
	void initDirs();

	void setupModel();

	std::tuple<cv::Mat, torch::Tensor> convertMatFileToTensor(const std::string& filename);

protected:
	virtual void readSpecialSettings() = 0;
	virtual std::shared_ptr<BaseModelImpl> createModel(std::string model_cfg) = 0;
	virtual void loadWeights(std::string model_cfg) = 0;
	virtual void setupSpecialObjects() = 0;

	virtual std::vector<torch::Tensor> postprocess(torch::Tensor preds) = 0;
	virtual void plotsResultToMat(cv::Mat src_image, std::string filename, std::vector<torch::Tensor> bboxes, torch::Tensor protos) = 0;
protected:
	std::string m_rootDir;
	std::string m_sourceDir;
	std::string m_saveDir;

	VariantConfigs m_args;
	int m_imageSize = 640;
	bool m_bScaleFill = false;
	bool m_bshowPred = true;
	
	torch::Device m_device = torch::Device(torch::kCPU);
	std::shared_ptr<BaseModelImpl> m_modelPtr = { nullptr };
};