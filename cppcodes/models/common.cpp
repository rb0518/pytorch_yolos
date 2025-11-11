#include <common.h>
#include <vector>
#include <glog/logging.h>
#include <cmath>

//=================  nnUpsampleImpl ================================

nnUpsampleImpl::nnUpsampleImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    set_params(in_channels, number, args);
}

void nnUpsampleImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    in_ch = in_channels;
    out_ch = in_channels;
    // args[0] None未用
    scale_size = static_cast<int64_t>(std::get<int>(args[1]));
    str_modetype = std::get<std::string>(args[2]);
    std::cout << scale_size << " " << str_modetype << std::endl;
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

#if 0
/*
* pytorch 原函数下注释为
   r"""Applies the Gaussian Error Linear Units function.

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is 'tanh', Gelu is estimated with:

    .. math:: \textx * ({GELU}(x) = 0.5 * 1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

    Args:
        approximate (str, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
   网上查资料
*/

torch::Tensor GELUImpl::forward(torch::Tensor x)
{
    //精度要求不高的近似解如下：
    //return 0.5 * x * (1 + torch::tanh(std::sqrt(2 / M_PI) * (x + 0.044715 * torch::pow(x, 3)));
    // 标准公式代码如下：
    return x * 0.5 * (1 + torch::erf(x * std::sqrt(2)));

}
#endif