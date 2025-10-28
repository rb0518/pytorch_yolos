#pragma once

#include <torch/torch.h>

#include "yaml_load.h"
#include "datasets.h"
#include "yolo.h"
#include "lambda_lr.h"

class BaseTrainer
{
public:
	BaseTrainer(std::string _root, VariantConfigs cfg_default);

	void init_device();
	void init_dirs();

	void setup_model();
	void load_pretrained();
	void freeze_modules();

	void init_dataloader();

	void setup_optimizer();
	void setup_scheduler();

	void resume_training();

	void do_train();
	std::vector<torch::Tensor> preprocess_batch(YoloCustomExample& batch);
	void do_warmup(int epoch, int ni, int nw, int nbs, int batch_size);

	virtual std::string refresh_batch_end_info(int epoch, int epochs, int idx, int num_targets,
		torch::Tensor& mloss, torch::Tensor& loss_items);
	virtual void on_train_epoch_over(int epoch);
	virtual void save_model() {};
	virtual void load_model() {};
protected:
	void build_optiomizer(int nc, std::string optim_name, 
		float _lr, float _momentum, float _decay, int iterations);
	void save_batch_sample_image(int epoch, int base_epoch, int start_epoch,
		torch::Tensor data, torch::Tensor targets);
protected:
	std::string root_dir;		// 传递根目录
	VariantConfigs args;

	std::string task_name;	// 

	std::string save_dir;	// 保存日志等文件路径，绝对路径
	std::string last_pt_file;
	std::string best_pt_file;
	std::string results_file;

	int start_epochs;
	int epochs;

	int accumulate = 1;

	torch::Device device = torch::Device(torch::kCPU);
	Model model{ nullptr };
	std::shared_ptr<torch::optim::Optimizer> optimizer;
	std::string optimizer_name;
	DataloaderBase* train_loader;

	std::shared_ptr<LambdaLR> scheduler;
};