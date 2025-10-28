#include <common.h>
#include <vector>
#include <glog/logging.h>
#include <cmath>

//=================  nnUpsampleImpl ================================
void nnUpsampleImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    in_ch = in_channels;
    out_ch = in_channels;
    // args[0] None未用
    scale_size = static_cast<int64_t>(std::get<int>(args[1]));
    str_modetype = std::get<std::string>(args[2]);
#if 0
    // 该代码在Linux下无错，但windows下报错
    auto str_size = std::get<std::string>(args[0]);
    if (str_size == "None")
        this->bneed_calcsize = true;
        
    auto options = torch::nn::UpsampleOptions().mode(torch::kNearest);
    // mode: torch::kNearest, don't set align_corners
    if(bneed_calcsize == false)
        options = options.scale_factor(std::vector<double>({ double(scale_size), double(scale_size) }));

    if (str_modetype == "Linear" || str_modetype == "linear")
    {
        LOG(ERROR) << "4D input, mode == linear error";
        options.mode(torch::kLinear).align_corners(false);
    }
    else if(str_modetype == "Bilinear" || str_modetype == "bilinear")
        options.mode(torch::kBilinear).align_corners(false);
    else if (str_modetype == "Bicubic" || str_modetype == "bicubic")
        options.mode(torch::kBicubic).align_corners(false);
    else if (str_modetype == "Trilinear" || str_modetype == "trilinear")
        options.mode(torch::kTrilinear).align_corners(false);
    upsample = torch::nn::Upsample(options);
    register_module("up", upsample);
#endif
}

torch::Tensor nnUpsampleImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
torch::Tensor nnUpsampleImpl::forward(torch::Tensor x)
{
    
    int H = x.sizes()[2];
    int W = x.sizes()[3];
#if 0
    if(bneed_calcsize)
        upsample->options.size(std::vector<int64_t>{H* scale_size, W* scale_size});

    auto ret = upsample->forward(x);
#else
    auto ret = torch::nn::functional::interpolate(x,
        torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>{H* scale_size, W* scale_size})  // 显式指定输出尺寸
        .mode(torch::kNearest));
#endif
    return ret;
}

void ContractImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    in_ch = in_channels;
    gain = std::get<int>(args[0]);
    out_ch = in_channels * gain * gain;
}

torch::Tensor ContractImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
torch::Tensor ContractImpl::forward(torch::Tensor x)
{
    auto b = x.size(0);  // (1, 64, 80, 80)
    auto c = x.size(1);
    auto h = x.size(2);
    auto w = x.size(3);

    auto x_ret = x.view({b, c, h/gain, gain, w/gain, gain});  // (1,64,40,2,40,2)
    x_ret = x_ret.permute({0, 3, 5, 1, 2, 4}).contiguous(); // x(1,2,2,64,40,40)
    return x_ret.view({b, c*gain*gain, h/gain, w/gain});   // (1, 256, 40, 40)
}

void ExpandImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    in_ch = in_channels;
    gain = std::get<int>(args[0]);
    out_ch = in_channels / (gain * gain);
}

torch::Tensor ExpandImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
torch::Tensor ExpandImpl::forward(torch::Tensor x)
{
    auto b = x.size(0);  // (1, 64, 80, 80)
    auto c = x.size(1);
    auto h = x.size(2);
    auto w = x.size(3);
    int gain_pow2 = gain * gain; 
    auto x_ret = x.view({b, gain, gain, c / gain_pow2, h, w});  // (1,2, 2, 16, 80, 80)
    x_ret = x_ret.permute({0, 3, 4, 1, 5, 2}).contiguous(); // x(1,16, 80,2,80, 2)
    return x_ret.view({b, c/gain_pow2, h * gain, w * gain});   // (1, 16, 160, 160)
}
