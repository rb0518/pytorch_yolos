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
	void do_train();

	virtual void setup_trainer() = 0;
	virtual void setup_model() = 0;
	virtual void init_dataloader() = 0;

	void get_args_settings();
	void check_resume();

	void init_device();
	void init_dirs();

	void load_pretrained();
	void freeze_modules();

	virtual void setup_optimizer();

	void setup_scheduler();
	void resume_training();

	void do_warmup(int epoch, int ni, int nw, int nbs, int batch_size);

	virtual std::string refresh_batch_end_info(int epoch, int epochs, int idx, int num_targets,
		torch::Tensor& loss, torch::Tensor& mloss, torch::Tensor& loss_items);
	virtual void on_train_epoch_over(int epoch);
	virtual void on_train_over();
	void save_model(int epoch);
	virtual void load_model() {};

public:	
	//	virtual void setup_model_from_pt(std::string ptfile) = 0;		// 从pt文件中读取上次训练相关设置：imgsz, model cfg, scale_id
	// virtual void load_weight(torch::serialize::InputArchive& in_archive) = 0;	// 从pt文件中载入weights, load

	virtual void write_envsets_to_pt(torch::serialize::OutputArchive& out_archive) = 0;	// 子类写各自的特色设置
	virtual void preprocess_batch(YoloCustomExample& batch) = 0;

protected:
	std::shared_ptr<torch::optim::Optimizer> build_optiomizer(int nc, std::string optim_name,
		float _lr, float _momentum, float _decay, int iterations);
	void save_batch_sample_image(int epoch, int base_epoch, int start_epoch,
		torch::Dict<std::string, torch::Tensor>& batch);
protected:
	std::string root_dir;		// 传递根目录
	VariantConfigs args;

	std::string task_name;	// 

	bool plots_flags = true;
	std::string model_file = "";
	bool b_resume = false;
	
	std::string save_dir;	// 保存日志等文件路径，绝对路径
	std::string last_pt_file;
	std::string best_pt_file;
	std::string results_file;

	bool resume_load_ok = false;

	int start_epochs;
	int epochs;

	torch::Tensor loss;
	torch::Tensor loss_items;

	int accumulate = 1;

	torch::Device device = torch::Device(torch::kCPU);
	
	std::shared_ptr<BaseModelImpl> model;
	std::shared_ptr<torch::optim::Optimizer> optimizer;

	std::string optimizer_name;
	std::shared_ptr<DataloaderBase> train_loader;
	std::shared_ptr<LambdaLR> scheduler;
};

class DetectionTrainer : public BaseTrainer
{
public:
	DetectionTrainer(std::string _root, VariantConfigs cfg_default)
		: BaseTrainer(_root, cfg_default){ }

	virtual void setup_trainer();
	virtual void setup_model();
	virtual void init_dataloader();

	virtual void write_envsets_to_pt(torch::serialize::OutputArchive& out_archive);	// 子类写各自的特色设置
	virtual void preprocess_batch(YoloCustomExample& batch);
};