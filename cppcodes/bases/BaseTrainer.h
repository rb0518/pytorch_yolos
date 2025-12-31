#pragma once

#include <torch/torch.h>

#include "lambda_lr.h"

#include "globaldef.h"
#include "BaseData.h"
#include "BaseModel.h"
#include "BaseLoss.h"

#include "BaseValidate.h"

/*
*	demo code:	
*   std::shared_ptr<BaseTrainder> trainer = std::make_shared<XXXTrainer>(root, cfg);
*	trainer->setupTrainer();
*   trainer->doTrainTask();		
*/

class BaseTrainer
{
public:
	BaseTrainer(VariantConfigs cfg_default, std::string root_dir);
	void setupTrainer();
	void doTrainTask();		// 执行训练

public:		
	virtual void getSpecialSettins() = 0;
	virtual void setupModel() = 0;
	virtual void setupDataloader() = 0;
	virtual void write_envsets_to_pt(torch::serialize::OutputArchive& out_archive) = 0;	// 子类写各自的特色设置
	virtual void preprocess_batch(VariantData& batch) = 0;
	virtual void specialTreatWhenEpochOver() = 0;		// 由子类继承做专项处理用		
	virtual void specialTreatWhenTrainover() = 0;
	virtual void createComputeLoss() = 0;
	virtual void createValidator() = 0;
protected:
	std::string m_rootDir;
	VariantConfigs m_args;

	bool m_bPlotsEnable = true;		// 是否需要plot输出
	bool m_bResume = false;

	std::string m_saveDir;	// 保存日志等文件路径，绝对路径
	torch::Device m_device = torch::Device(torch::kCPU);

	std::string m_lastPtFile;
	std::string m_bestPtFile;
	std::string m_resultsFile;

	std::string m_modelFilename = "";	

	bool m_bVal;
protected:
	void getArgsSettings();
	void checkResumeSettings();

	void initDevice();
	void initDirs();

	void loadPretrained();
	void freezeModules();

	virtual void setupOptimizer();

	void setupScheduler();
	void resumeLastTraining();

	void doWarmup(int epoch, int ni, int nw, int nbs, int batch_size);

	void on_train_epoch_over(int epoch);
	void on_train_over();

	void save_model(int epoch);
	virtual void load_model() {};

protected:
	std::string refresh_batch_end_info(int epoch, int epochs, int idx, int num_targets);
	std::shared_ptr<torch::optim::Optimizer> build_optiomizer(int nc, std::string optim_name,
		float _lr, float _momentum, float _decay, int iterations);
	void save_batch_sample_image(int epoch, int base_epoch, int start_epoch,
		VariantData& batch);
protected:

//	bool resume_load_ok = false;

	int start_epochs;
	int epochs;
	int accumulate = 1;

	
	std::shared_ptr<BaseModelImpl> model;
	std::shared_ptr<torch::optim::Optimizer> optimizer;
	std::string optimizer_name;
	std::shared_ptr<LambdaLR> scheduler;
	std::shared_ptr<BaseDataloader> train_loader;
	std::shared_ptr<BaseLoss> m_compute_loss;

	std::shared_ptr<BaseValidate> m_validator = nullptr;
};