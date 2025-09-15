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
//    std::cout << "ConvImple: " << c1 << " " << c2 << " " << k << " " << s << " " << p << std::endl;        
//    std::cout << "ConvImple: " << in_ch << " " << out_ch << " " << k_size << " " << stride << " " << padding << std::endl;
   /*
   conv = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, k).stride(s).padding(padding)),
        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c2)),
        torch::nn::SiLU()
    );
    */
    // bias(false)能将weights从150M缩减到28M，后续采用half，还能进一步减小weights的尺寸
    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, k).stride(s).padding(padding).bias(false).groups(1).dilation(1));
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c2));
    this->silu = torch::nn::SiLU();  //本可以用torch::silu直接用，但为了与pytorch代码保持一致，按这样操作
    // bn->register_parameter("running_mean", bn->running_mean);
    // bn->register_parameter("running_var", bn->running_var);
    register_module("conv", conv);
    register_module("bn", bn);
}

void ConvImpl::fuse_conv_and_bn(bool fused_ /*= true*/)
{
    if (fused_)
    {
        fused = fused_;
        this->fusedconv = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_ch, out_ch, k_size).stride(stride).
            padding(padding).
            bias(true).groups(1).dilation(1));

        auto w_conv = conv->weight.clone().view({ out_ch, -1 });
        auto w_bn = torch::diag(bn->weight.div(torch::sqrt(1e-5 + bn->running_var)));
        fusedconv->weight.copy_(torch::mm(w_bn, w_conv).view({ fusedconv->weight.sizes() }));

        auto b_conv = torch::zeros({ conv->weight.size(0) }).to(conv->weight.device());
        auto b_bn = bn->bias - bn->weight.mul(bn->running_mean).div(torch::sqrt(bn->running_var + 1e-5));
        fusedconv->bias.copy_(torch::mm(w_bn, b_conv.reshape({ -1, 1 })).reshape({ -1 }) + b_bn);

        replace_module("conv", fusedconv);

        for (auto& param : fusedconv->named_parameters())
            param->set_requires_grad(false);
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
#if 0 
    if (fused || is_training() == false)
    {
        
        this->fusedconv = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_ch, out_ch, k_size).stride(stride).
            padding(padding).
            bias(true).groups(1).dilation(1));

        fusedconv->to(x.device());

        auto w_conv = conv->weight.clone().view({ out_ch, -1 });
        auto w_bn = torch::diag(bn->weight.div(torch::sqrt(1e-5 + bn->running_var)));
        fusedconv->weight.copy_(torch::mm(w_bn, w_conv).view({ fusedconv->weight.sizes() }));

        auto b_conv = torch::zeros({ conv->weight.size(0) }).to(conv->weight.device());
        auto b_bn = bn->bias - bn->weight.mul(bn->running_mean).div(torch::sqrt(bn->running_var + 1e-5));
        fusedconv->bias.copy_(torch::mm(w_bn, b_conv.reshape({ -1, 1 })).reshape({ -1 }) + b_bn);

        for (auto& param : fusedconv->parameters())
            param.set_requires_grad(false);
        
        return silu->forward(conv->forward(x));
    }
    return silu->forward(bn->forward(conv->forward(x)));
#endif   
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
    /*
    std::vector<torch::Tensor> out(3);
    out[0] = cv1->forward(x);
    out[1] = cv2->forward(out[0]);
    if(b_shortcut)
    {
//        std::cout << "run in shortcut type" << std::endl;
        out[2] = x + out[1];
        return out[2];
    }    
//    std::cout << "run not with shortcut" << std::endl;
    return out[1];
    */
}

torch::Tensor BottleneckImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}

void C3Impl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    check_args(args);
    Init_Modules(in_channels, std::get<int>(args[0]), number,
        std::get<bool>(args[1]), float(std::get<int>(args[2])));
}

C3Impl::C3Impl(int c1, int c2, int n, bool shortcut, float e)
{
    Init_Modules(c1, c2, n, shortcut, e);
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
         bottlenecks.push_back(Bottleneck(c_, c_, shortcut, 1.0));
         register_module("m-"+std::to_string(i), bottlenecks[i]);

     }

//    std::cout << "C3 c1" << c1 << " c2 " << c2 << " shortcut: " << shortcut << std::endl;
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
    std::vector<torch::Tensor> b_outs(number);
    //std::cout << "number: " << number << " input size: " << x.sizes() << std::endl;
    torch::Tensor c1_out = cv1->forward(x);
    torch::Tensor c2_out = cv2->forward(x);

    ///std::cout << "c1_out " << c1_out.sizes() << " c2_out " << c2_out.sizes() << std::endl;
    for (int i = 0; i < number; i++)
    {
        if(i == 0)
            b_outs[i] = bottlenecks[i]->forward(c1_out);
        else
            b_outs[i] = bottlenecks[i]->forward(b_outs[i-1]);
    }
    //std::cout << "bottleneck number :" << b_outs[number-1].sizes() << std::endl;            
    torch::Tensor cat_out = torch::cat({ b_outs[number - 1], c2_out}, 1);
    return cv3->forward(cat_out);
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
    //std::cout << "SPPF forward input: " << x.sizes() << std::endl;
    torch::Tensor x1 = cv1->forward(x);
    //std::cout << "SPPF forward input cv1: " << x1.sizes() << std::endl;
    torch::Tensor y1 = m->forward(x1);
    //std::cout << "SPPF forward m1 : " << y1.sizes() << std::endl;
    torch::Tensor y2 = m->forward(y1);
    //std::cout << "SPPF forward m2 : " << y2.sizes() << std::endl;
    torch::Tensor y3 = m->forward(y2);
    //std::cout << "SPPF forward m3 : " << y3.sizes() << std::endl;
    torch::Tensor cat_out = torch::cat({x1, y1, y2, y3}, 1);
    //std::cout << "SPPF forward cat : " << cat_out.sizes() << std::endl;

    torch::Tensor ret = cv2->forward(cat_out);
    //std::cout << "SPPF forward cv2 : " << ret.sizes() << std::endl;
    return ret;
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
        m->push_back(register_module("m-" + std::to_string(i),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(k[i])
                .stride(1).padding(p))));
    }
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
