#pragma once

#include <torch/torch.h>

#include "yaml_load.h"

// libtorch中没有tensorboard,要的话，只能自己写或者集成python，暂时不做
void train(const std::string& _root,  
	VariantConfigs& hyp,
	VariantConfigs& opt,
	torch::Device& device,
	const std::string& jit_script_file = ""
);	

// 简单快速修改代码来测试Segment训练相关内容，只看损失函数，后续等完成对应val部分代码后再来合并两种train
void train_seg(const std::string& _root,  
	VariantConfigs& hyp,
	VariantConfigs& opt,
	torch::Device& device,
	const std::string& jit_script_file = ""
);	