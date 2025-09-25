#include "lambda_lr.h"

LambdaLR::LambdaLR(std::shared_ptr<torch::optim::Optimizer> optimizer,
    float start,
    float end,
    int step_size,
    bool linear_lr,
    int last_step)
    : LRScheduler(*optimizer), y1(start), y2(end), step_size_(step_size), linear_lr_(linear_lr),
    last_step_(last_step), _optimizer(optimizer)
{
    initial_lrs = get_current_lrs();
}

std::vector<double> LambdaLR::get_lrs()
{
    last_step_ = std::max(0, last_step_ + 1);
    cur_lrs = get_current_lrs();
    for (int i = 0; i < cur_lrs.size(); i++)
    {
        if (linear_lr_)
            cur_lrs[i] = fn_linear_lr(last_step_);
        else
            cur_lrs[i] = fn_one_cycle(last_step_);
        cur_lrs[i] = cur_lrs[i] * initial_lrs[i];
    }
    return cur_lrs;
}

double LambdaLR::fn_linear_lr(int x)
{
    auto ret = (1.0 - double(x) / double(step_size_ - 1)) * (1.0 - double(y2)) + double(y2);
    return ret;
}
double LambdaLR::fn_one_cycle(int x)
{
    auto ret = ((1.0 - cos(double(x) * M_PI / double(step_size_))) / 2.0) * double(y2 - y1) + double(y1);
    return ret;
}

void LambdaLR::step()
{
    //也可以完全不用，只需要完成get_lrs虚函数，由基类step自动处理
     cur_lrs = get_lrs();
     auto param_groups = _optimizer->param_groups();
     for(int i = 0; i < _optimizer->param_groups().size(); i++)
     {
         if(_optimizer->param_groups()[i].has_options())
             _optimizer->param_groups()[i].options().set_lr(cur_lrs[i]);
     }
}
