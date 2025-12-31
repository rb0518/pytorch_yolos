#pragma once

#include <torch/torch.h>

#include "yaml_load.h"
#include "yolo.h"
#include "lambda_lr.h"

#include "BaseTrainer.h"

class DetectionTrainer : public BaseTrainer
{
public:
	DetectionTrainer(VariantConfigs cfg_default, std::string root_dir)
		: BaseTrainer(cfg_default, root_dir){ }

	virtual void getSpecialSettins(){};		
	virtual void setupModel();
	virtual void setupDataloader();

	virtual void write_envsets_to_pt(torch::serialize::OutputArchive& out_archive);	// 子类写各自的特色设置
	virtual void preprocess_batch(VariantData& batch);

	virtual void specialTreatWhenEpochOver();		// 由子类继承做专项处理用		
	virtual void specialTreatWhenTrainover();
	virtual void createComputeLoss();

	virtual void createValidator();
};