
#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>

class GradScaler {
public:
    explicit GradScaler(
        float init_scale = 32.0f,
        float growth_factor = 2.0f,
        float backoff_factor = 0.5f,
        int growth_interval = 2000,
        bool enabled = true
    );
    
    // 缩放输出张量
    torch::Tensor scale(const torch::Tensor& outputs);
    std::vector<torch::Tensor> scale(const std::vector<torch::Tensor>& outputs);
    
    // 更新缩放因子
    void update();
    
    // 获取当前缩放因子
    torch::Tensor get_scale() const { return scale_; }
    
    bool is_enabled() const { return enabled_; }
    
private:
    torch::Tensor scale_;
    torch::Tensor growth_tracker_;
    bool enabled_;
    float growth_factor_;
    float backoff_factor_;
    int growth_interval_;
    
    void lazy_init_scale(const torch::Device& device);
};
