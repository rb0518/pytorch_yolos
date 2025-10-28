#include "conv.h"

#include <vector>
#include <glog/logging.h>
#include <cmath>
#include <algorithm> // 包含 std::gcd

inline int autopad(int k, std::optional<int> p = std::nullopt, int d = 1)
{
    if(d > 1)
        k = d*(k-1) + 1;
    
    if (p == std::nullopt)
    {
        p = static_cast<int>(std::trunc(float(k - 1) / 2.0));
    }

    if (!p.has_value())
    {
        std::cout << "autopad p have_value == false: return 0\n";
        return 0;
    }

    if (*p < 0)
    {
        std::cout << "*********** autopad p < 0 *************** "<< (*p) << " \n";
    }

    return std::max(*p, 0);
}

inline torch::nn::AnyModule createActivation(std::string act_type = "SiLU")
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

// ------------------ start Conv -----------------------------
void ConvImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args) 
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
        std::cout << "ConvImpl args number error. \n";
}

ConvImpl::ConvImpl(int c1, int c2, int k, int s, std::optional<int> p, int g , int d , std::string act)
{
    Init_Modules(c1, c2, k, s, p, g, d, act);
}

void ConvImpl::Init_Modules(int c1, int c2, int k, int s, std::optional<int> p, int g, int d, std::string act)
{
    in_ch = c1;
    out_ch = c2;
    k_size = k;
    stride = s;
    padding = autopad(k, p, 1);

    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, k).stride(s).padding(padding).bias(false).groups(g).dilation(d));
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c2));
    register_module("conv", conv);
    register_module("bn", bn);

    _act = createActivation(act);
    register_module("act", _act.ptr());
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
            torch::nn::Conv2dOptions(in_ch, out_ch, k_size).stride(stride).
            padding(padding).
            bias(true).groups(1).dilation(1));
        fusedconv->to(conv->parameters().begin()->device());
        for (auto& param : fusedconv->named_parameters())
            param->set_requires_grad(false);

        auto w_conv = conv->weight.clone().view({ out_ch, -1 });
        auto w_bn = torch::diag(bn->weight.div(torch::sqrt(1e-5 + bn->running_var)));
        fusedconv->weight.copy_(torch::mm(w_bn, w_conv).view({ fusedconv->weight.sizes() }));

        auto b_conv = torch::zeros({ conv->weight.size(0) }).to(conv->weight.device());
        auto b_bn = bn->bias - bn->weight.mul(bn->running_mean).div(torch::sqrt(bn->running_var + 1e-5));
        fusedconv->bias.copy_(torch::mm(w_bn, b_conv.reshape({ -1, 1 })).reshape({ -1 }) + b_bn);

        replace_module("conv", fusedconv);
        unregister_module("bn");
        
        bfirst = false;
    }
    else
    {
        replace_module("conv", conv);
    }
}

void ConvImpl::check_args(std::vector<arg_complex>& args)
{
}

torch::Tensor ConvImpl::forward(torch::Tensor x)
{
    if(fused == false)
        return _act.forward(bn->forward(conv->forward(x)));

    return _act.forward(fusedconv->forward(x));
}

torch::Tensor ConvImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
// ------------------ End Conv -----------------------------

// ------------------ start Conv2 -----------------------------
void Conv2Impl::set_params(int in_channels, int number, std::vector<arg_complex>& args) 
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
        std::cout << "Conv2Impl args number error. \n";
}

Conv2Impl::Conv2Impl(int c1, int c2, int k, int s, std::optional<int> p, int g, int d, std::string act)
{
    Init_Modules(c1, c2, k, s, p, g, d, act);
}

void Conv2Impl::Init_Modules(int c1, int c2, int k, int s, std::optional<int> p, int g, int d,  std::string act)
{
    in_ch = c1;
    out_ch = c2;
    k_size = k;
    stride = s;
    padding = autopad(k, p, d);

    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, k).stride(s).padding(padding).bias(false).groups(g).dilation(d));
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c2));
    register_module("conv", conv);
    register_module("bn", bn);

    _act = createActivation(act);
    int k_cv2 = 1;
    auto padding_cv2 = autopad(k_cv2, p, d);
    cv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, 1).stride(s).padding(padding).bias(false).groups(1).dilation(d));
    register_module("cv2", cv2);
}

void Conv2Impl::fuse_conv_and_bn(bool fused_ /*= true*/)
{
    if (fused_)
    {
        fused = fused_;

        auto w = torch::zeros_like(conv->weight.data());
        int w_h = w.size(2);
        int w_w = w.size(3);

        w.index_put_({ torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(w_h, w_h + 1),
                      torch::indexing::Slice(w_w, w_w + 1) },
            cv2->weight.data().clone());
        conv->weight.data().add_(w);
        unregister_module("cv2");
        
        bfirst = false;
    }
    else
    {
        replace_module("conv", conv);
    }
}

void Conv2Impl::check_args(std::vector<arg_complex>& args)
{}

torch::Tensor Conv2Impl::forward(torch::Tensor x)
{
    if(fused == false)
        return _act.forward(bn->forward(conv->forward(x) + cv2->forward(x)));   

    return _act.forward(bn->forward(conv->forward(x)));  
}

torch::Tensor Conv2Impl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
// ------------------ End Conv2 -----------------------------

// ------------------ start DWConv -----------------------------
void DWConvImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    check_args(args);       // check input args size
    Init_Modules(in_channels,
        std::get<int>(args[0])
    );
}

DWConvImpl::DWConvImpl(int c1, int c2, int k, int s, int d, std::string act)
{
    Init_Modules(c1, c2, k, s, d, act);
}

void DWConvImpl::Init_Modules(int c1, int c2, int k /*= 1*/, int s /*= 1*/, int d /*= 1*/, std::string act/* = "True"*/)
{
    in_ch = c1;
    out_ch = c2;
    k_size = k;
    stride = s;
    padding = autopad(k, std::nullopt, d);
    int gcd_g = std::gcd(c1, c2);
    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, k)
            .stride(s)
            .padding(padding)
            .bias(false)
            .groups(gcd_g)
            .dilation(d));

    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c2));

    _act = createActivation(act);

    register_module("conv", conv);
    register_module("bn", bn);
}

void DWConvImpl::fuse_conv_and_bn(bool fused_ /*= true*/)
{
    if (fused_)
    {
        fused = fused_;
        this->fusedconv = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_ch, out_ch, k_size).stride(stride).
            padding(padding).
            bias(true).groups(1).dilation(1));
        fusedconv->to(conv->parameters().begin()->device());
        for (auto& param : fusedconv->named_parameters())
            param->set_requires_grad(false);

        auto w_conv = conv->weight.clone().view({ out_ch, -1 });
        auto w_bn = torch::diag(bn->weight.div(torch::sqrt(1e-5 + bn->running_var)));
        fusedconv->weight.copy_(torch::mm(w_bn, w_conv).view({ fusedconv->weight.sizes() }));

        auto b_conv = torch::zeros({ conv->weight.size(0) }).to(conv->weight.device());
        auto b_bn = bn->bias - bn->weight.mul(bn->running_mean).div(torch::sqrt(bn->running_var + 1e-5));
        fusedconv->bias.copy_(torch::mm(w_bn, b_conv.reshape({ -1, 1 })).reshape({ -1 }) + b_bn);

        replace_module("conv", fusedconv);
        unregister_module("bn");

        bfirst = false;
    }
    else
    {
        replace_module("conv", conv);
    }
}

void DWConvImpl::check_args(std::vector<arg_complex>& args)
{
    int argsize = args.size();
    if (argsize == 2)    // 插入in后只有3个，说明还差两个
    {
        arg_complex tmp_s = 1;
        args.push_back(tmp_s);
        arg_complex tmp_p = -1;
        args.push_back(tmp_p);
    }
    if (argsize == 3)
    {
        arg_complex tmp_p = -1;
        args.push_back(tmp_p);
    }
}


torch::Tensor DWConvImpl::forward(torch::Tensor x)
{
    if (fused == false)
        return _act.forward(bn->forward(conv->forward(x)));

    return _act.forward(fusedconv->forward(x));
}

torch::Tensor DWConvImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
// ------------------ End DWConv -----------------------------

// ------------------ start LightConv -----------------------------
void LightConvImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args) 
{
    check_args(args);       // check input args size
    Init_Modules(in_channels, 
        std::get<int>(args[0]),
        std::get<int>(args[1]),
        std::get<std::string>(args[2]));
}

LightConvImpl::LightConvImpl(int c1, int c2, int k, std::string act) 
{
    Init_Modules(c1, c2, k, act);
}

void LightConvImpl::Init_Modules(int c1, int c2, int k, std::string act)
{
    in_ch = c1;
    out_ch = c2;
    k_size = k;
    conv1 = Conv(c1, c2, 1, 1, std::nullopt, 1, 1, "False");
    conv2 = DWConv(c2, c2, k, 1, 1, act);
    register_module("conv1", conv1);
    register_module("conv2", conv2);
}

void LightConvImpl::check_args(std::vector<arg_complex>& args)
{
    int argsize = args.size();
    if(argsize == 1)    
    {
        arg_complex tmp_k = 1;
        args.push_back(tmp_k);
        arg_complex tmp_act = "ReLU";
        args.push_back(tmp_act);
    }
    if(argsize == 2)
    {
        arg_complex tmp_act = "ReLU";
        args.push_back(tmp_act);
    }
}


torch::Tensor LightConvImpl::forward(torch::Tensor x)
{
    return conv2->forward(conv1->forward(x));
}

torch::Tensor LightConvImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
// ------------------ End LightConv -----------------------------

//------------------- Start Focus ------------------------------
FocusImpl::FocusImpl(int c1, int c2, int k/* = 1*/, int s/* = 1*/,
    std::optional<int> p/* = std::nullopt*/, int g /*= 1 */, std::string act /*= "True"*/)
{
    Init_Modules(c1, c2, k, s, p, g, act);
}

void FocusImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
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
        std::cout << "FocusImpl args number error. \n";

}

void FocusImpl::Init_Modules(int c1, int c2, int k/* = 1*/, int s/* = 1*/,
    std::optional<int> p/* = std::nullopt*/, int g /*= 1*/, std::string act /*= "True"*/)
{
    in_ch = c1;
    out_ch = c2;
    k_size = k;
    stride = s;

    conv = Conv(c1*4, c2, k, s, p, g, 1, act);

    register_module("conv", conv);
}

void FocusImpl::check_args(std::vector<arg_complex>& args)
{}

torch::Tensor FocusImpl::forward(torch::Tensor x)
{
/*
    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))
*/
    // [x[..., ::2, ::2]
    auto x_0 = x.index({"...",
        torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2),
        torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2)
        });
    // x[..., 1::2, ::2]
    auto x_1 = x.index({"...",
        torch::indexing::Slice(1, torch::indexing::None, 2),
        torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2)
        });
    // x[..., ::2, 1::2]
    auto x_2 = x.index({"...",
        torch::indexing::Slice(torch::indexing::None, torch::indexing::None, 2),
        torch::indexing::Slice(1, torch::indexing::None, 2)
        });
    // x[..., 1::2, 1::2]
    auto x_3 = x.index({"...",
        torch::indexing::Slice(1, torch::indexing::None, 2),
        torch::indexing::Slice(1, torch::indexing::None, 2)
        });

    // 沿通道维度拼接
    return conv->forward(torch::cat({x_0, x_1, x_2, x_3}, 1));
}

torch::Tensor FocusImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
// ------------------ End Focus -----------------------------

//------------------- Start Concat ------------------------------
void ConcatImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    in_ch = in_channels;
    out_ch = in_channels;
    dimen = std::get<int>(args[0]);
}

torch::Tensor ConcatImpl::forward(std::vector<torch::Tensor> x)
{
//    std::cout << "Concat  forward input" << x[0].sizes() << " - " << x[1].sizes() << std::endl;
    torch::Tensor ret = torch::cat(x, dimen);
//    std::cout << "Concat forward output: " << ret.sizes() << std::endl;
    return ret;
}

torch::Tensor ConcatImpl::forward(torch::Tensor x)
{
    LOG(ERROR) << "This module only accept mult-input";
    return x;
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

GhostConvImpl::GhostConvImpl(int c1, int c2, int k, int s, int g, std::string act)
{
    Init_Modules(c1, c2, k, s, g, act);
}

torch::Tensor GhostConvImpl::forward(std::vector<torch::Tensor> x)
{
    auto y = cv1->forward(x);
    return torch::cat({y, cv2->forward(y)}, 1);
}

torch::Tensor GhostConvImpl::forward(torch::Tensor x)
{
    LOG(ERROR) << "This module only accept mult-input";
    return x;
}

void GhostConvImpl::Init_Modules(int c1, int c2, int k, int s, int g, std::string act)
{
    int in_ch = c1;
    int out_ch = c2;
    int k_size = k;
    int stride = s;
    int d = 1;
    int c_ = static_cast<int>(std::trunc(float(c2) / 2.0));

    cv1 = Conv(c1, c_, k, s, std::nullopt, g, d, act);
    cv2 = Conv(c_, c_, 5, 1, std::nullopt, c_, d, act);
    register_module("cv1", cv1);
    register_module("cv2", cv2);
}

// ------------------ End GhostConv -----------------------------

// ------------------ start ConvTranspose -----------------------------
void ConvTransposeImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    check_args(args);       // check input args size
    Init_Modules(in_channels,
        std::get<int>(args[0]),
        std::get<int>(args[1])
    );
}

ConvTransposeImpl::ConvTransposeImpl(int c1, int c2, int k, int s, int p0, bool bn, std::string act)
{
    Init_Modules(c1, c2, k, s, p0, bn, act);
}

void ConvTransposeImpl::Init_Modules(int c1, int c2, int k , int s, int p0, bool bn, std::string act)
{
    in_ch = c1;
    out_ch = c2;
    k_size = k;
    stride = s;

    conv_transpose = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(c1, c2, k)
        .stride(s).padding(p0).bias(!bn));

    if (bn)
        _bn = std::make_shared<torch::nn::BatchNorm2dImpl>(c2);
    else
        _bn = std::make_shared<torch::nn::IdentityImpl>();
    register_module("bn", _bn.ptr());

    _act = createActivation(act);
    register_module("act", _act.ptr());
}

void ConvTransposeImpl::check_args(std::vector<arg_complex>& args)
{
    int argsize = args.size();
    if (argsize == 2)    // 插入in后只有3个，说明还差两个
    {
        arg_complex tmp_s = 1;
        args.push_back(tmp_s);
        arg_complex tmp_p = -1;
        args.push_back(tmp_p);
    }
    if (argsize == 3)
    {
        arg_complex tmp_p = -1;
        args.push_back(tmp_p);
    }
}

torch::Tensor ConvTransposeImpl::forward(torch::Tensor x)
{
    if (fused == false)
        return _act.forward(_bn.forward(conv_transpose->forward(x)));

    return _act.forward(conv_transpose->forward(x));
}

torch::Tensor ConvTransposeImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
// ------------------ End Conv -----------------------------