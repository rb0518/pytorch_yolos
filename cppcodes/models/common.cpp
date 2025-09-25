#include <common.h>
#include <vector>

#include <glog/logging.h>
#include <cmath>

void ConvImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args) 
{
    check_args(args);       // check input args size
    Init_Modules(in_channels, 
        std::get<int>(args[0]),
        std::get<int>(args[1]),
        std::get<int>(args[2]),
        std::get<int>(args[3])
        );
}

ConvImpl::ConvImpl(int c1, int c2, int k, int s, int p) 
{
    Init_Modules(c1, c2, k, s, p);
}

void ConvImpl::Init_Modules(int c1, int c2, int k, int s, int p)
{
    in_ch = c1;
    out_ch = c2;
    k_size = k;
    stride = s;
    padding = p;
    if(p < 0)
        padding = static_cast<int>(std::trunc(float(k-1)/2.0));
    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, k).stride(s).padding(padding).bias(false).groups(1).dilation(1));
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c2));
    this->silu = torch::nn::SiLU();  //本可以用torch::silu直接用，但为了与pytorch代码保持一致，按这样操作
    // bn->register_parameter("running_mean", bn->running_mean);
    // bn->register_parameter("running_var", bn->running_var);
    register_module("conv", conv);
    register_module("bn", bn);
}

/*
    测试了去年BatchNorm2d，修改fusedconv的weight和bias，同等训练没有正确推理
    输出
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

            for (auto& param : fusedconv->named_parameters())
                param->set_requires_grad(false);

            auto w_conv = conv->weight.clone().view({ out_ch, -1 });
            auto w_bn = torch::diag(bn->weight.div(torch::sqrt(1e-5 + bn->running_var)));
            fusedconv->weight.copy_(torch::mm(w_bn, w_conv).view({ fusedconv->weight.sizes() }));

            auto b_conv = torch::zeros({ conv->weight.size(0) }).to(conv->weight.device());
            auto b_bn = bn->bias - bn->weight.mul(bn->running_mean).div(torch::sqrt(bn->running_var + 1e-5));
            fusedconv->bias.copy_(torch::mm(w_bn, b_conv.reshape({ -1, 1 })).reshape({ -1 }) + b_bn);

            replace_module("conv", fusedconv);
            
            bfirst = false;
    }
    else
    {
        replace_module("conv", conv);
    }
}

void ConvImpl::check_args(std::vector<arg_complex>& args)
{
    int argsize = args.size();
    if(argsize == 2)    // 插入in后只有3个，说明还差两个
    {
        arg_complex tmp_s = 1;
        args.push_back(tmp_s);
        arg_complex tmp_p = -1;
        args.push_back(tmp_p);
    }
    if(argsize == 3)
    {
        arg_complex tmp_p = -1;
        args.push_back(tmp_p);
    }
}


torch::Tensor ConvImpl::forward(torch::Tensor x)
{
    return silu->forward(bn->forward(conv->forward(x)));   
}

torch::Tensor ConvImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}

BottleneckImpl::BottleneckImpl(int c1, int c2, bool shortcut, float e)
{
    in_ch = c1;
    out_ch = c2;
    expansion = e;

    int c_ = int(float(c2) * e);
    cv1 = Conv(c1, c_, 1, 1, -1);
    cv2 = Conv(c_, c2, 3, 1, -1);
    register_module("cv1", cv1);
    register_module("cv2", cv2);

    b_shortcut = (shortcut && c1 == c2);
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x)
{
    if(b_shortcut)
        return x + cv2->forward(cv1->forward(x));
    return cv2->forward(cv1->forward(x));
}
//=================  BottleneckCSPImpl ================================
void BottleneckCSPImpl::check_args(std::vector<arg_complex>& args)
{

}

void BottleneckCSPImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    check_args(args);
    Init_Modules(in_channels, std::get<int>(args[0]), number,
        std::get<bool>(args[1]), float(std::get<int>(args[2])));
}

void BottleneckCSPImpl::Init_Modules(int c1, int c2, int n /*= 1*/, bool shortcut_ 
        /*= true*/, int g /*=1*/, float e /*= 0.5*/)
{
    in_ch = c1;
    out_ch = c2;
    number = n;
    expansion = e;
    int c_ = int(float(c2) * e);
    cv1 = Conv(c1, c_, 1, 1, -1);
    cv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c_, 1).stride(1).bias(false));
    cv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(c_, c_, 1).stride(1).bias(false));
    cv4 = Conv(2*c_, c2, 1, 1, -1);
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(2*c_));
    act = torch::nn::SiLU();

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    register_module("cv4", cv4);
    register_module("bn", bn);
    for(int i = 0; i < n; i++){
         m->push_back(Bottleneck(c_, c_, shortcut_, 1.0));
    }
    register_module("m", m);
}

torch::Tensor BottleneckCSPImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
torch::Tensor BottleneckCSPImpl::forward(torch::Tensor x)
{
    torch::Tensor y1 = cv3->forward(m->forward(cv1->forward(x)));
    torch::Tensor y2 = cv2->forward(x);
    return cv4->forward(act->forward(bn->forward(torch::cat({y1, y2}, 1))));
}

//=================  C3Impl ================================
void C3Impl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    check_args(args);
    Init_Modules(in_channels, std::get<int>(args[0]), number,
        std::get<bool>(args[1]), float(std::get<int>(args[2])));
}

void C3Impl::Init_Modules(int c1, int c2, int n, bool shortcut, float e)
{
    in_ch = c1;
    out_ch = c2;
    number = n;
    expansion = e;
    int c_ = int(float(c2) * 0.5);
    cv1 = Conv(c1, c_, 1, 1, -1);
    cv2 = Conv(c1, c_, 1, 1, -1);
    cv3 = Conv(2*c_, c2, 1, 1, -1);


    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);

     for(int i = 0; i < n; i++){
         m->push_back(Bottleneck(c_, c_, shortcut, 1.0));
     }
     register_module("m", m);
}

void C3Impl::check_args(std::vector<arg_complex>& args)
{
    //args.size = 4
    int argsize = args.size();
    if(argsize == 1)
    {
        arg_complex tmp_shortcut = true;
        args.push_back(tmp_shortcut);
        arg_complex tmp_e = 1; // 暂时未做修改，因为python代码未管这个输入值
        args.push_back(tmp_e);
    }
    if(argsize == 2)
    {
        arg_complex tmp_e = 1; // 暂时未做修改，因为python代码未管这个输入值
        args.push_back(tmp_e);
    }
}

torch::Tensor C3Impl::forward(torch::Tensor x)
{
    return cv3->forward(torch::cat({m->forward(cv1->forward(x)), cv2->forward(x)}, 1));
}

torch::Tensor C3Impl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
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

#include <math.h>
//=================  SPPFImpl ================================
void SPPFImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    Init_Modules(in_channels, std::get<int>(args[0]), std::get<int>(args[1]));
}

void SPPFImpl::Init_Modules(int c1, int c2, int k)
{
    in_ch = c1;
    out_ch = c2;
    int c_ = static_cast<int>(std::trunc(float(c1)/2.0));
    int p = static_cast<int>(std::trunc(k/2));

    //std::cout << "SPPF: " << c1 << " " << c_ << " p " << p << std::endl;
    cv1 = Conv(c1, c_, 1, 1, -1);
    cv2 = Conv(c_ * 4, c2, 1, 1, -1);
    m = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(k).stride(1).padding(p));
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("m", m);
}

torch::Tensor SPPFImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
torch::Tensor SPPFImpl::forward(torch::Tensor x)
{
    torch::Tensor x1 = cv1->forward(x);
    torch::Tensor y1 = m->forward(x1);
    torch::Tensor y2 = m->forward(y1);
    return cv2->forward(torch::cat({x1, y1, y2, m->forward(y2)}, 1));
}
//------------------- SPP -------------------------------
// SPP 参数形式为 [1024, [5, 9, 13]]  SPPF: [1024, 5]
// 暂时将参数写死，后续修改
void SPPImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    Init_Modules(in_channels, std::get<int>(args[0]));
}

void SPPImpl::Init_Modules(int c1, int c2)
{
    in_ch = c1;
    out_ch = c2;
    int c_ = static_cast<int>(std::trunc(float(c1)/2.0));

    //std::cout << "SPPF: " << c1 << " " << c_ << " p " << p << std::endl;
    cv1 = register_module("cv1", Conv(c1, c_, 1, 1, -1));
    cv2 = register_module("cv2", Conv(c_ * (k.size() + 1), c2, 1, 1, -1));
    for(int i = 0; i < k.size(); i++)
    {
        int p = static_cast<int>(std::trunc(k[i] / 2));
        m->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(k[i])
                .stride(1).padding(p)));
    }
    register_module("m", m);
}

torch::Tensor SPPImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
torch::Tensor SPPImpl::forward(torch::Tensor x)
{
    x = cv1->forward(x);
    std::vector<torch::Tensor> outputs = { x };

    for (auto& layer : *m) 
    {
        outputs.push_back(layer->as<torch::nn::MaxPool2d>()->forward(x));
    }

    return cv2->forward(torch::cat(outputs, 1));
}

//------------------- Focus ------------------------------
FocusImpl::FocusImpl(int c1, int c2, int k, int s, int p)
{
    Init_Modules(c1, c2, k, s, p);
}

void FocusImpl::set_params(int inchannels, int number, std::vector<arg_complex>& args)
{
    check_args(args);       // check input args size
    Init_Modules(inchannels, 
        std::get<int>(args[0]),
        std::get<int>(args[1]),
        std::get<int>(args[2]),
        std::get<int>(args[3])
        );
}

void FocusImpl::Init_Modules(int c1, int c2, int k, int s, int p)
{
    in_ch = c1;
    out_ch = c2;
    k_size = k;
    stride = s;
    padding = p;
    if(p < 0)
        padding = static_cast<int>(std::trunc(float(k-1)/2.0));

    conv = Conv(c1*4, c2, k, s, p);

    register_module("conv", conv);
}

void FocusImpl::check_args(std::vector<arg_complex>& args)
{
    int argsize = args.size();
    if(argsize == 2)    // 插入in后只有3个，说明还差两个k, p
    {
        arg_complex tmp_s = 1;
        args.push_back(tmp_s);
        arg_complex tmp_p = -1;
        args.push_back(tmp_p);
    }
    if(argsize == 3)
    {
        arg_complex tmp_p = -1;
        args.push_back(tmp_p);
    }
}

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
// =========================== ProtoImpl ====================
ProtoImpl::ProtoImpl(int c1, int c_, int c2)
{
    in_ch = c1;
    out_ch = c2;
    cv1 = Conv(c1, c_, 3, 1, -1);
    cv2 = Conv(c_, c_, 3, 1, -1);
    cv3 = Conv(c_, c2, 1, 1, -1);
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    torch::nn::UpsampleOptions options = torch::nn::UpsampleOptions().mode(torch::kNearest).scale_factor(std::vector<double>({ double(2), double(2) }));
    upsample = torch::nn::Upsample(options);
}

torch::Tensor ProtoImpl::forward(torch::Tensor x)
{
#if 0
    torch::Tensor x1 = cv1->forward(x);
    int H = x1.sizes()[2];
    int W = x1.sizes()[3];
    auto x2 = torch::nn::functional::interpolate(x1,
        torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>{H*2, W*2})  // 显式指定输出尺寸
        .mode(torch::kNearest));
    return cv3->forward(cv2->forward(x2));
#else
    return cv3->forward(cv2->forward(upsample->forward(cv1->forward(x))));
#endif    
}

