#pragma once

#include <torch/torch.h>
#include "yolo.h"
#include "yaml_load.h"

// dev_set: "CUDA"、"CPU"、"GPU" or "0"...
torch::Device get_device(const std::string& dev_set);

std::tuple<std::shared_ptr<torch::optim::Optimizer>, int, float> smart_optimizer(std::shared_ptr<ModelImpl> ptr_model, 
                                    VariantConfigs& args);


void save_checkpoint(std::string filename, torch::nn::Module* model,
    std::shared_ptr<torch::optim::Optimizer> optimizer, int epoch);
                     
// 没有deepcopy，同时不想重构Model来支持Module.clone() 
class ModelEMA {
public:
    ModelEMA(std::shared_ptr<Model> ptr_model, float decay = 0.9999f, int updates = 0);
    void update(torch::nn::Module& model);

    float decay_function(int x)
    {
        return decay_ * (1 - std::exp(-x / 2000.0f));
    }

public:
    std::shared_ptr<Model> ema_model_;
    float decay_;
    int updates_ = 0;
};    