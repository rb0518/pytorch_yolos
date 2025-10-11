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
