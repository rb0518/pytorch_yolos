#pragma once

#include <torch/torch.h>

// libtorch中没有预封装LambdaLR类，只能对应pytorch lr_scheduler.py中代码，通过继承LRScheduler类
class LambdaLR : public torch::optim::LRScheduler
{
public:
    LambdaLR(
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        float start,
        float end,
        int step_size,
        bool linear_lr = false,
        int last_step = -1);
    virtual ~LambdaLR() = default;
    
    std::vector<double> get_curstep_lrs()
    {
        return get_current_lrs();
    }
    void step();

private:
    std::vector<double> get_lrs() override;

    float y1;
    float y2;
    bool linear_lr_;
    int step_size_;
    int last_step_;

    std::vector<double> initial_lrs;
    std::vector<double> cur_lrs;

    double fn_linear_lr(int x);
    double fn_one_cycle(int x);

    std::shared_ptr<torch::optim::Optimizer> _optimizer;
};
