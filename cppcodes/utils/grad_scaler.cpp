#include "grad_scaler.h"
#include <stdexcept>

GradScaler::GradScaler(
    float init_scale,
    float growth_factor,
    float backoff_factor,
    int growth_interval,
    bool enabled
) : enabled_(enabled),
    growth_factor_(growth_factor),
    backoff_factor_(backoff_factor),
    growth_interval_(growth_interval) {
    
    if (enabled_) {
        scale_ = torch::tensor(init_scale);
        growth_tracker_ = torch::tensor(0);
    }
}

torch::Tensor GradScaler::scale(const torch::Tensor& outputs) {
    if (!enabled_) {
        return outputs;
    }
    
    lazy_init_scale(outputs.device());
    return outputs * scale_.to(outputs.device());
}

std::vector<torch::Tensor> GradScaler::scale(const std::vector<torch::Tensor>& outputs) {
    if (!enabled_) {
        return outputs;
    }
    
    std::vector<torch::Tensor> scaled_outputs;
    for (const auto& tensor : outputs) {
        scaled_outputs.push_back(scale(tensor));
    }
    return scaled_outputs;
}

void GradScaler::update() {
    if (!enabled_) {
        return;
    }
    
    auto growth_tracker = growth_tracker_.item<int>();
    growth_tracker++;
    
    if (growth_tracker >= growth_interval_) {
        auto new_scale = scale_.item<float>() * growth_factor_;
        scale_ = torch::tensor(new_scale);
        growth_tracker = 0;
    }
    
    growth_tracker_ = torch::tensor(growth_tracker);
}

void GradScaler::lazy_init_scale(const torch::Device& device) {
    if (!scale_.defined()) {
        scale_ = torch::tensor(65536.0f).to(device);
        growth_tracker_ = torch::tensor(0).to(device);
    }
}
