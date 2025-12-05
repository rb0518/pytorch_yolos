#include "conv.h"

#include <vector>
#include <glog/logging.h>
#include <cmath>
#include <algorithm> // 包含 std::gcd

#include <typeinfo>
#include <utility>

#include "common.h"
#include <experimental/optional>

int autopad(int k, std::optional<int> p /*= std::nullopt*/, int d /*= 1*/)
{
    if(d > 1)
        k = d*(k-1) + 1;
    
    if (p == std::nullopt)
        p = (k - 1) / 2;
#ifdef _SHOW_DEBUG_STEP_INFO
    if (!p.has_value())
    {
        std::cout << "autopad p have_value == false: return 0\n";
        return 0;
    }

    if (*p < 0)
    {
        std::cout << "*********** autopad p < 0 *************** "<< (*p) << " \n";
    }
#endif
    return std::max(*p, 0);
}

torch::nn::AnyModule createActivation(std::string act_type = "SiLU")
{
    torch::nn::AnyModule act;
    if(act_type == "False")
        act = std::make_shared<torch::nn::IdentityImpl>();
    else if(act_type == "ReLU")
        act = std::make_shared<torch::nn::ReLUImpl>();
    else if(act_type == "True")
        act = std::make_shared<torch::nn::SiLUImpl>();
    else
    {
        act = std::make_shared<torch::nn::SiLUImpl>();
        LOG(ERROR)  << "Init not define activation type: " << act_type;       
    }

    return act;
}

// ===================== start Conv =========================================

ConvImpl::ConvImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    set_params(in_channels, number, args);
}

ConvImpl::ConvImpl(int c1, int c2, int k, int s, std::optional<int> p, int g , int d , std::string act)
{
    Init_Modules(c1, c2, k, s, p, g, d, act);
}

void ConvImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args) 
{
    Conf_Params defs = get_defaults();
    make_param_from_args<Conf_Params>(defs, args);

    if(args.size() < 4) // 保持 p = nullopt
        Init_Modules(in_channels, std::get<0>(defs), std::get<1>(defs), std::get<2>(defs));
    else
        Init_Modules(in_channels, std::get<0>(defs), std::get<1>(defs), std::get<2>(defs), 
            std::get<3>(defs),
            std::get<4>(defs), std::get<5>(defs), std::get<6>(defs));
}

void ConvImpl::Init_Modules(int c1, int c2, int k, int s, std::optional<int> p, int g, int d, std::string act)
{
    _in_channel = c1;
    _out_channels = c2;
    _kernel_size = k;
    _stride = s;
    _padding = autopad(k, p, 1);
    _groups = g;
    _dilation = d;
    _act = act;

    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, k).stride(s).padding(*_padding).bias(false).groups(g).dilation(d));
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c2));
    active_fn = createActivation(act);
    register_module("conv", conv);
    register_module("bn", bn);
    register_module("act", active_fn.ptr());
}

/*
    测试了去掉BatchNorm2d，修改fusedconv的weight和bias，相同训练过程没有正确推理
    输出，但转入jit.script权重，能够正确推理
*/
void ConvImpl::fuse_conv_and_bn(bool fused_ /*= true*/)
{
    if (fused_)
    {
        fused = fused_;
        this->fusedconv = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(_in_channel, _out_channels, _kernel_size).stride(_stride).
            padding(*_padding).bias(true).groups(_groups).dilation(_dilation));
        fusedconv->to(conv->parameters().begin()->device());
        for (auto& param : fusedconv->named_parameters())
            param->set_requires_grad(false);

        auto w_conv = conv->weight.clone().view({ _out_channels, -1 });
        auto w_bn = torch::diag(bn->weight.div(torch::sqrt(1e-5 + bn->running_var)));
        fusedconv->weight.copy_(torch::mm(w_bn, w_conv).view({ fusedconv->weight.sizes() }));

        auto b_conv = torch::zeros({ conv->weight.size(0) }).to(conv->weight.device());
        auto b_bn = bn->bias - bn->weight.mul(bn->running_mean).div(torch::sqrt(bn->running_var + 1e-5));
        fusedconv->bias.copy_(torch::mm(w_bn, b_conv.reshape({ -1, 1 })).reshape({ -1 }) + b_bn);

        replace_module("conv", fusedconv);
        unregister_module("bn");
    }
    else
    {
        replace_module("conv", conv);
    }
}

torch::Tensor ConvImpl::forward(torch::Tensor x)
{
    std::vector<torch::Tensor> rets(3);

    if(fused)
        return active_fn.forward(fusedconv->forward(x));

    rets[0] = conv->forward(x);
    rets[1] = bn->forward(rets[0]);
    rets[2] = active_fn.forward(rets[1]);

    //return active_fn.forward(bn->forward(conv->forward(x)));
    return rets[2];
}
// ------------------------------ End Conv -----------------------------

// ================================= start Conv2 ==================================

Conv2Impl::Conv2Impl(int c1, int c2, int k, int s, std::optional<int> p, int g, int d, std::string act)
    : ConvImpl(c1, c2, k, s, p, g, d, act)
{
    int k_cv2 = 1;
    auto padding_cv2 = autopad(k_cv2, p, d);
    cv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, 1).stride(s).padding(padding_cv2).bias(false).groups(g).dilation(d));
    
    register_module("cv2", cv2);
}
/*
void Conv2Impl::Init_Modules(int c1, int c2, int k, int s, std::optional<int> p, int g, int d,  std::string act)
{
    _in_channel = c1;
    _out_channels = c2;
    _kernel_size = k;
    _stride = s;
    _padding = autopad(k, p, d);
    _groups = g;
    _dilation = d;
    _act = act;

    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, k).stride(s).padding(*_padding).bias(false).groups(g).dilation(d));
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c2));
    active_fn = createActivation(act);

    register_module("conv", conv);
    register_module("bn", bn);
    register_module("act", active_fn.ptr());

    int k_cv2 = 1;
    auto padding_cv2 = autopad(k_cv2, p, d);
    cv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, 1).stride(s).padding(padding_cv2).bias(false).groups(g).dilation(d));
    
    register_module("cv2", cv2);
}
*/
torch::Tensor Conv2Impl::forward(torch::Tensor x)
{
    if(fused)
        return active_fn.forward(bn->forward(conv->forward(x)));

    return active_fn.forward(bn->forward(conv->forward(x) + cv2->forward(x)));   
}
// ------------------ End Conv2 -----------------------------

// ------------------ start DWConv -----------------------------
DWConvImpl::DWConvImpl(int c1, int c2, int k, int s, int d, std::string act)
    : ConvImpl(c1, c2, k, s, std::nullopt, std::gcd(c1, c2), d, act)
{    
}
// ------------------ End DWConv -----------------------------

// ------------------ start LightConv -----------------------------
LightConvImpl::LightConvImpl(int c1, int c2, int k, std::string act) 
{
    conv1 = Conv(c1, c2, 1, 1, std::nullopt, 1, 1, "False");
    conv2 = DWConv(c2, c2, k, 1, 1, act);
    register_module("conv1", conv1);
    register_module("conv2", conv2);
}

torch::Tensor LightConvImpl::forward(torch::Tensor x)
{
    return conv2->forward(conv1->forward(x));
}
// ------------------ End LightConv -----------------------------
// 
//------------------- Start Focus ------------------------------

FocusImpl::FocusImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    set_params(in_channels, number, args);
}

FocusImpl::FocusImpl(int c1, int c2, int k/* = 1*/, int s/* = 1*/,
    std::optional<int> p/* = std::nullopt*/, int g /*= 1 */, std::string act /*= "True"*/)
{
    Init_Modules(c1, c2, k, s, p, g, act);
}

void FocusImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    int args_size = args.size();
    if (args_size == 1)
        Init_Modules(in_channels, std::get<int>(args[0]));
    else if (args_size == 2)
        Init_Modules(in_channels, std::get<int>(args[0]), std::get<int>(args[1]));
    else if (args_size == 3)
        Init_Modules(in_channels, std::get<int>(args[0]), std::get<int>(args[1]), std::get<int>(args[2]));
    else
        std::cout << "FocusImpl args number error: " << args_size;

}

void FocusImpl::Init_Modules(int c1, int c2, int k/* = 1*/, int s/* = 1*/,
    std::optional<int> p/* = std::nullopt*/, int g /*= 1*/, std::string act /*= "True"*/)
{
    _out_channels = c2;

    conv = Conv(c1 * 4, c2, k, s, p, g, 1, act);
    register_module("conv", conv);
}


torch::Tensor FocusImpl::forward(torch::Tensor x)
{
/*
    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))
*/
    // [x[..., ::2, ::2]
    return conv->forward(torch::cat({
            x.index({"...", torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2),
                            torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2)}),
            x.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2),
                            torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2)}),
            x.index({"...", torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2),
                            torch::indexing::Slice(1, torch::indexing::None, 2)}),
            x.index({"...", torch::indexing::Slice(1, torch::indexing::None, 2),
                            torch::indexing::Slice(1, torch::indexing::None, 2)})
            },1));
}
// ------------------ End Focus -----------------------------
// 
//------------------- Start Concat ------------------------------

ConcatImpl::ConcatImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    _out_channels = 0;
    set_params(in_channels, number, args);
}

void ConcatImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    _dimen = std::get<int>(args[0]);
}

// int ConcatImpl::get_outchannels()
// {
//     return 0;   // 该类返回是输入拼接后的维度，设定返回0，由外部程序计算
// }

torch::Tensor ConcatImpl::forward(std::vector<torch::Tensor> x)
{
    return torch::cat(x, _dimen);
}
// ------------------ End Concat -----------------------------

//------------------- Start GhostConv ------------------------------

void GhostConvImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    int argsize = args.size();
    if (argsize == 1)
        Init_Modules(in_channels, std::get<int>(args[0]));
    else if (argsize == 2)
        Init_Modules(in_channels, std::get<int>(args[0]), std::get<int>(args[1]));
    else if (argsize == 3)
        Init_Modules(in_channels, std::get<int>(args[0]), std::get<int>(args[1]), std::get<int>(args[2]));
    else if (argsize == 4)
        Init_Modules(in_channels, std::get<int>(args[0]), std::get<int>(args[1]), std::get<int>(args[2]), std::get<int>(args[3]));
    else
        std::cout << "GhostConvImpl args number error. \n";
}

GhostConvImpl::GhostConvImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    set_params(in_channels, number, args);
}

GhostConvImpl::GhostConvImpl(int c1, int c2, int k, int s, int g, std::string act)
{
    Init_Modules(c1, c2, k, s, g, act);
}

torch::Tensor GhostConvImpl::forward(torch::Tensor x)
{
    auto y = cv1->forward(x);
    return torch::cat({y, cv2->forward(y)}, 1);
}

void GhostConvImpl::Init_Modules(int c1, int c2, int k, int s, int g, std::string act)
{
    _out_channels = c2;

    int c_ = c2 / 2; // static_cast<int>(std::trunc(float(c2) / 2.0));

    cv1 = Conv(c1, c_, k, s, std::nullopt, g, 1, act);
    cv2 = Conv(c_, c_, 5, 1, std::nullopt, c_, 1, act);
    register_module("cv1", cv1);
    register_module("cv2", cv2);
}
// ------------------ End GhostConv -----------------------------

// ------------------ start ConvTranspose -----------------------------
void ConvTransposeImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    int args_size = args.size();
    if(args_size == 4)
        Init_Modules(in_channels, std::get<int>(args[0]),
            std::get<int>(args[1]), std::get<int>(args[2]), std::get<int>(args[3]));
    else
    {
        LOG(ERROR) << "ConvTransposeImpl number of args wrong(4): " << args_size;
    }
}

ConvTransposeImpl::ConvTransposeImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    set_params(in_channels, number, args);
}

ConvTransposeImpl::ConvTransposeImpl(int c1, int c2, int k, int s, int p, bool bn, std::string act)
{
    Init_Modules(c1, c2, k, s, p, bn, act);
}

void ConvTransposeImpl::Init_Modules(int c1, int c2, int k , int s, int p0, bool bn, std::string act)
{
    _out_channels = c2;

    conv_transpose = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(c1, c2, k)
        .stride(s).padding(p0).bias(!bn));

    if (bn)
        anymodule_bn = std::make_shared<torch::nn::BatchNorm2dImpl>(c2);
    else
        anymodule_bn = std::make_shared<torch::nn::IdentityImpl>();

    anymodule_act = createActivation(act);

    register_module("conv_transpose", conv_transpose);
    register_module("bn", anymodule_bn.ptr());
    register_module("act", anymodule_act.ptr());
}

torch::Tensor ConvTransposeImpl::forward(torch::Tensor x)
{
    return anymodule_act.forward(anymodule_bn.forward(conv_transpose->forward(x)));
}
// ------------------ End ConvTransposeImpl -----------------------------

// ------------------- Start DSConv ------------------------

DSConvImpl::DSConvImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    set_params(in_channels, number, args);
}

DSConvImpl::DSConvImpl(int c_in, int c_out, int k, int s, std::optional<int> p, int d, bool bias)
{
    Init_Modules(c_in, c_out, k, s, p, d, bias);
}

void DSConvImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    int args_cout = args.size();
    if (args_cout == 3)
    {
        Init_Modules(in_channels, std::get<int>(args[0]),
            std::get<int>(args[1]),
            std::get<int>(args[2]));
    }
    else {
        LOG(ERROR) << "number of DWConv args wrong: " << args_cout;
        Init_Modules(in_channels, std::get<int>(args[0]));
    }
}

torch::Tensor DSConvImpl::forward(torch::Tensor x)
{
    return act->forward(bn->forward(pw->forward(dw->forward(x))));
};

void DSConvImpl::Init_Modules(int c_in, int c_out, int k, int s, 
    std::optional<int> p , int d, bool bias)
{
    _out_channels = c_out;

    if (p == std::nullopt)
        p = autopad(k, p, d);

#ifdef _SHOW_DEBUG_STEP_INFO
    std::cout << "p == nullopt calc: " << padding << std::endl;
    std::cout << c_in << " " << c_out << " " << s << std::endl; 
#endif

    dw = torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_in, k)
        .stride(s).padding(*p).groups(c_in).bias(bias));
    pw = torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, 1)
        .stride(1).padding(0).bias(bias));
    bn = torch::nn::BatchNorm2d(c_out);
    act = torch::nn::SiLU();   

    register_module("dw", dw);
    register_module("pw", pw);
    register_module("bn", bn);
    register_module("act", act);
}

// ------------------- End DSConv --------------------------


ChannelAttentionImpl::ChannelAttentionImpl(int channels)
{
    _pool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 1, 1}));
    _fc = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 1).stride(1).padding(0).bias(true));
    _act = torch::nn::Sigmoid();

    register_module("pool", _pool);
    register_module("fc", _fc);
    register_module("act", _act);
}

torch::Tensor ChannelAttentionImpl::forward(torch::Tensor x)
{
    return x * _act->forward(_fc->forward(_pool->forward(x)));
}

// ------------------- End ChannelAttention --------------------------

SpatialAttentionImpl::SpatialAttentionImpl(int kernel_size)
{
    if (kernel_size == 7 || kernel_size == 3)
    {
        int padding = kernel_size == 7 ? 3 : 1;
        cv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 1, kernel_size)
            .padding(padding).bias(false));
        act = torch::nn::Sigmoid();
        register_module("cv1", cv1);
        register_module("act", act);
    }
    else
        LOG(ERROR) << "Kernel size muse be 7 or 7";
}

torch::Tensor SpatialAttentionImpl::forward(torch::Tensor x)
{
    auto x_mean = torch::mean(x, 1, true);
    auto [x_max_v, x_max_i] = torch::max(x, 1, true);

    return x * act->forward(cv1->forward(torch::cat({ x_mean, x_max_v }, 1)));
}
// ------------------- End SpatialAttention --------------------------

CBAMImpl::CBAMImpl(int c1, int kernel_size)
{
    channel_attention = ChannelAttention(c1);
    spatial_attention = SpatialAttention(kernel_size);

    register_module("channel_attention", channel_attention);
    register_module("spatial_attention", spatial_attention);
}

torch::Tensor CBAMImpl::forward(torch::Tensor x)
{
    return spatial_attention->forward(channel_attention->forward(x));
}
// ------------------- End CBAM --------------------------