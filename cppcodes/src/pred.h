#pragma once

#include "BasePredict.h"

class DetectPredict : public BasePredict 
{
public:
	explicit DetectPredict(VariantConfigs& args_input, std::string root_dir);

protected:
	virtual void readSpecialSettings()  override;
	virtual std::shared_ptr<BaseModelImpl> createModel(std::string model_cfg)  override;
	virtual void loadWeights(std::string model_cfg)  override;
	virtual void setupSpecialObjects()  override;

	virtual std::vector<torch::Tensor> postprocess(torch::Tensor preds)  override;
	virtual void plotsResultToMat(cv::Mat src_image, std::string filename, std::vector<torch::Tensor> bboxes, torch::Tensor protos)  override;
};