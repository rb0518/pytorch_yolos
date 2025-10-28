#include "block.h"
#include "common.h"

//------------------------------ Start DFL ----------------------------
DFLImpl::DFLImpl(int _c1)
{
    c1 = _c1;
    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, 1, 1).bias(false));
    conv->weight.requires_grad_(false);
    auto x = torch::arange(c1, torch::kFloat);
    conv->weight.data() = x.view({ 1, c1, 1, 1 });
    register_module("conv", conv);
}

torch::Tensor DFLImpl::forward(torch::Tensor x)
{
    // b, _, a = x.shape #batch, channels, anchors
    int b = x.size(0);
    int a = x.size(2);
    return conv->forward(x.view({ b, 4, c1, a }).transpose(2, 1).softmax(1)).view({ b, 4, a });
}
//------------------------------ End DFL ----------------------------

//------------------------------ Start Proto ----------------------------
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
    torch::nn::ConvTranspose2dOptions options = torch::nn::ConvTranspose2dOptions(
        c_, c_, 2).stride(2).padding(0).bias(true);
    upsample = torch::nn::ConvTranspose2d(options);
    register_module("upsample", upsample);
}

torch::Tensor ProtoImpl::forward(torch::Tensor x)
{
    //std::cout << "Proto input " << x.sizes() << std::endl;
    return cv3->forward(cv2->forward(upsample->forward(cv1->forward(x))));
}
//------------------------------ End Proto ----------------------------

//------------------------------ Start SPP ----------------------------
// SPP 参数形式为 [1024, [5, 9, 13]]  SPPF: [1024, 5]
// 暂时将参数写死，后续修改arg_complex, 能够读取vector<int>
void SPPImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    Init_Modules(in_channels, std::get<int>(args[0]));
}

void SPPImpl::Init_Modules(int c1, int c2, std::vector<int> _k)
{
    in_ch = c1;
    out_ch = c2;
    int c_ = static_cast<int>(std::trunc(float(c1) / 2.0));
    k = _k;
    //std::cout << "SPPF: " << c1 << " " << c_ << " p " << p << std::endl;
    cv1 = register_module("cv1", Conv(c1, c_, 1, 1));
    cv2 = register_module("cv2", Conv(c_ * (k.size() + 1), c2, 1, 1));
    for (int i = 0; i < k.size(); i++)
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
//------------------------------ End SPP ----------------------------

//============================== SPPFImpl ===========================
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
    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv(c_ * 4, c2, 1, 1);
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
//------------------------------ End SPPF ----------------------------

//============================== Start C2f ============================
C2fImpl::C2fImpl(int c1, int c2, int n, bool shortcut, int g, float e)
{
    Init_Modules(c1, c2, n, shortcut, g, e);
}

void C2fImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    int argsize = args.size();
    if (argsize == 1)
        Init_Modules(in_channels, std::get<int>(args[0]), number);
    else if (argsize == 2)
        Init_Modules(in_channels, std::get<int>(args[0]), number,
            std::get<bool>(args[1]));
    else
        std::cout << "C2fImpl arg number wrong.\n";
}

void C2fImpl::Init_Modules(int c1, int c2, int n, bool shortcut, int g,  float e)
{
    in_ch = c1;
    out_ch = c2;
    number = n;
    expansion = e;
    c = int(float(c2) * e);
    cv1 = Conv(c1, 2 * c, 1, 1);
    cv2 = Conv((2 + n) * c, c2, 1);


    register_module("cv1", cv1);
    register_module("cv2", cv2);

    for (int i = 0; i < n; i++) {
        m->push_back(Bottleneck(c, c, shortcut, g, std::make_tuple(3, 3), 1.0f));
    }
    register_module("m", m);
}

void C2fImpl::check_args(std::vector<arg_complex>& args)
{
}

torch::Tensor C2fImpl::forward(torch::Tensor x)
{
    auto y = cv1->forward(x).chunk(2, 1);
    for (int i = 0; i < number; i++)
    {
        auto tmp = y[y.size() - 1]; // 取队列最后一个
        auto mi = m[i]->as<Bottleneck>();
        auto r = mi->forward(tmp);
        y.push_back(r);
    }
    return cv2->forward(torch::cat(y, 1));
}

torch::Tensor C2fImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}

torch::Tensor C2fImpl::forward_split(torch::Tensor x)
{
    auto y_split = cv1->forward(x).split({ c, c }, 1);
    std::vector<torch::Tensor> y;
    y.emplace_back(y_split[0]);
    y.emplace_back(y_split[1]);
    for (int i = 0; i < number; i++)
    {
        auto tmp = y[y.size() - 1]; // 取队列最后一个
        auto mi = m[i]->as<Bottleneck>();
        auto r = mi->forward(tmp);
        y.emplace_back(r);
    }
    return cv2->forward(torch::cat(y, 1));
}
//------------------------------ End C2f ----------------------------

//============================== Start C3k2 ============================
C3k2Impl::C3k2Impl(int c1, int c2, int n, bool c3k, float e, int g, bool shortcut)
{
    Init_Modules(c1, c2, n, c3k, e, g, shortcut);
}

void C3k2Impl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    int argsize = args.size();
    if (argsize == 1)
        Init_Modules(in_channels, std::get<int>(args[0]), number);
    else if (argsize == 2)
        Init_Modules(in_channels, std::get<int>(args[0]), number,
            std::get<bool>(args[1]));
    else if (argsize == 3)
        Init_Modules(in_channels, std::get<int>(args[0]), number,
            std::get<bool>(args[1]),
            std::get<float>(args[2])
            );
    else
        std::cout << "C2fImpl arg number wrong.\n";
}

void C3k2Impl::Init_Modules(int c1, int c2, int n, bool c3k, float e, int g, bool shortcut)
{
    in_ch = c1;
    out_ch = c2;
    number = n;
    expansion = e;
    use_c3k = c3k;

    c = int(float(c2) * e);
    cv1 = Conv(c1, 2 * c, 1, 1);
    cv2 = Conv((2 + n) * c, c2, 1);
    //std::cout << "C3K2Impl: c1 " << c1 << " c2 " << c2 << " c " << c << " e " << e << std::endl;

    register_module("cv1", cv1);
    register_module("cv2", cv2);

    for (int i = 0; i < n; i++) {
        if(use_c3k)
            m->push_back(C3k(c, c, 2, shortcut, g));
        else
            m->push_back(Bottleneck(c, c, shortcut, g));
    }
    register_module("m", m);
}

void C3k2Impl::check_args(std::vector<arg_complex>& args)
{
}

torch::Tensor C3k2Impl::forward(torch::Tensor x)
{
    auto y = cv1->forward(x).chunk(2, 1);
    for (int i = 0; i < number; i++)
    {
        auto tmp = y[y.size() - 1]; // 取队列最后一个
        if(use_c3k)
        {
            auto mi = m[i]->as<C3k>();
            auto r = mi->forward(tmp);
            y.emplace_back(r);
        }
        else
        {
            auto mi = m[i]->as<Bottleneck>();
            auto r = mi->forward(tmp);
            y.emplace_back(r);
        }
    }
    return cv2->forward(torch::cat(y, 1));
}

torch::Tensor C3k2Impl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}

torch::Tensor C3k2Impl::forward_split(torch::Tensor x)
{
    auto y_split = cv1->forward(x).split({ c, c }, 1);
    std::vector<torch::Tensor> y;
    y.emplace_back(y_split[0]);
    y.emplace_back(y_split[1]);
    for (int i = 0; i < number; i++)
    {
        auto tmp = y[y.size() - 1]; // 取队列最后一个
        if (use_c3k)
        {
            auto mi = m[i]->as<C3k>();
            auto r = mi->forward(tmp);
            y.emplace_back(r);
        }
        else
        {
            auto mi = m[i]->as<Bottleneck>();
            auto r = mi->forward(tmp);
            y.emplace_back(r);
        }
    }
    return cv2->forward(torch::cat(y, 1));
}
//------------------------------ End C3k2 ----------------------------

//============================== Start C3k ============================
C3kImpl::C3kImpl(int c1, int c2, int n, bool shortcut, int g, float e, int k_)
{
    int c_ = int(float(c2) * e);
    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv(c1, c_, 1, 1);
    cv3 = Conv(2*c_, c2, 1, 1);


    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    
    for(int i = 0; i < n; i++){
        m->push_back(*(Bottleneck(c_, c_, shortcut, g, std::make_tuple(k_, k_), 1.0)));
    }
    register_module("m", m);
    //std::cout << "init C3kImpl ok...\n";
}
//------------------------------ End C3k ----------------------------

//============================== Start C3 ============================
C3Impl::C3Impl(int c1, int c2, int n, bool shortcut, int g, float e)
{
    Init_Modules(c1, c2, n, shortcut, e);
}

void C3Impl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    check_args(args);
    Init_Modules(in_channels, std::get<int>(args[0]), number,
        std::get<bool>(args[1]), std::get<float>(args[2]));
}

void C3Impl::Init_Modules(int c1, int c2, int n, bool shortcut, int g, float e)
{
    in_ch = c1;
    out_ch = c2;
    number = n;
    expansion = e;
    int c_ = int(float(c2) * e);
    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv(c1, c_, 1, 1);
    cv3 = Conv(2*c_, c2, 1, 1);


    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    std::vector<std::tuple<int, int>> ks;
    ks.emplace_back(std::make_tuple(1, 1));
    ks.emplace_back(std::make_tuple(3, 3));
    for(int i = 0; i < n; i++){
        m->push_back(Bottleneck(c_, c_, shortcut, 1, ks[i % ks.size()], 1.0));
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
        arg_complex tmp_e = 0.5f; // 暂时未做修改，因为python代码未管这个输入值
        args.push_back(tmp_e);
    }
    if(argsize == 2)
    {
        arg_complex tmp_e = 0.5f; // 暂时未做修改，因为python代码未管这个输入值
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
//------------------------------ End C3 ----------------------------

//============================== Start C3x ============================
C3xImpl::C3xImpl(int c1, int c2, int n, bool shortcut, float e)
{
    Init_Modules(c1, c2, n, shortcut, e);
}

void C3xImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    check_args(args);
    Init_Modules(in_channels, std::get<int>(args[0]), number,
        std::get<bool>(args[1]));
}

void C3xImpl::Init_Modules(int c1, int c2, int n, bool shortcut, float e)
{
    in_ch = c1;
    out_ch = c2;
    number = n;
    expansion = e;
    int c_ = int(float(c2) * e);
    cv1 = Conv(c1, c_, 1, 1, -1);
    cv2 = Conv(c1, c_, 1, 1, -1);
    cv3 = Conv(2*c_, c2, 1, 1, -1);


    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    std::vector<std::tuple<int, int>> ks;
    ks.emplace_back(std::make_tuple(1, 3));
    ks.emplace_back(std::make_tuple(3, 1));
    for(int i = 0; i < n; i++){
        m->push_back(Bottleneck(c_, c_, shortcut, 1, ks[i % ks.size()], 1.0));
    }
    register_module("m", m);
}

void C3xImpl::check_args(std::vector<arg_complex>& args)
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

torch::Tensor C3xImpl::forward(torch::Tensor x)
{
    return cv3->forward(torch::cat({m->forward(cv1->forward(x)), cv2->forward(x)}, 1));
}

torch::Tensor C3xImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
//------------------------------ End C3x ----------------------------

//============================== Start Bottleneck ============================
//BottleneckImpl::BottleneckImpl(int c1, int c2, bool shortcut, 
//            int g, std::tuple<int, int> k,  
//            float e)
BottleneckImpl::BottleneckImpl(int c1, int c2, bool shortcut/* = true*/,
    int g/* = 1*/, std::tuple<int, int> k/* = std::make_tuple(3, 3)*/,
    float e/* = 0.5f*/)
{
    in_ch = c1;
    out_ch = c2;
    expansion = e;

    int c_ = int(float(c2) * e);
    auto [k0, k1] = k;
    cv1 = Conv(c1, c_, k0, 1);
    cv2 = Conv(c_, c2, k1, 1, std::nullopt, g);
    register_module("cv1", cv1);
    register_module("cv2", cv2);

    add_flag = (shortcut && c1 == c2);
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x)
{
    if(add_flag)
        return x + cv2->forward(cv1->forward(x));
    return cv2->forward(cv1->forward(x));
}
//------------------------------ End Bottleneck ----------------------------


//============================== Start A2C2f ============================
class A2C2f_2ABlock : public torch::nn::Module 
{
public:
    A2C2f_2ABlock(int c1, int c2, float mlp_r, int area)
    {
        a1 = ABlock(c1, c2, mlp_r, area);
        a2 = ABlock(c1, c2, mlp_r, area);
        register_module("m0", a1);
        register_module("m1", a2);
    }
    torch::Tensor forward(torch::Tensor x)
    {
        return a2->forward(a1->forward(x));
    }
public:
    ABlock a1{ nullptr };
    ABlock a2{ nullptr };
};


A2C2fImpl::A2C2fImpl(int c1, int c2, int n, bool a2 , int area, bool residual, 
        float mlp_ratio, float e, int g, bool shortcut)
{
    Init_Modules(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut);
}

void A2C2fImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    // [-1, 4, A2C2f, [512, True, 4/1/-1]] ==> c2 = 512,  n = 4,  a2 = True, area = 4/1/-1
    check_args(args);
    int argsize = args.size();
    if(argsize == 3)
    {
//        std::cout << "Init width 3 args.\n";
        Init_Modules(in_channels, 
            std::get<int>(args[0]), 
            number,
            std::get<bool>(args[1]), 
            std::get<int>(args[2]));
    } else{
        std::cout << "Init width 5 args.\n";
        Init_Modules(in_channels, 
            std::get<int>(args[0]), 
            number,
            std::get<bool>(args[1]), 
            std::get<int>(args[2]),
            std::get<bool>(args[3]), 
            std::get<int>(args[4])
        );
    }
}

void A2C2fImpl::Init_Modules(int c1, int c2, int n, bool a2 , int area, bool residual, 
        float mlp_ratio, float e, int g, bool shortcut)
{
    in_ch = c1;
    out_ch = c2;
    number = n;

    a2_ = a2;
    residual_ = residual;

    int c_ = int(float(c2) * e);

    // std::cout << "c1 " << c1 << " c2 " << c2 << " n " << n << " a2 " << a2
    //     << " area: " << area << " residual " << residual << " mlp_r " << mlp_ratio
    //     << " e " << e << " g " << g << " shortcut " << shortcut << "\n";

    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv((1+n)*c_, c2, 1);

    register_module("cv1", cv1);
    register_module("cv2", cv2);

    if (a2 && residual)
    {
        gamma = 0.1f * torch::ones(c2);
        //gamma.requires_grad_(true);
        register_parameter("gamma", gamma);
    }

    for(int i = 0; i < n ; i++)
    {
        if(a2)
        {
//            std::cout << "a2 is true " << i << std::endl;
            m->push_back(A2C2f_2ABlock(c_, c_/32, mlp_ratio, area));
        }
        else{
//            std::cout << "a2 is false " << i << std::endl;
            m->push_back(C3k(c_, c_, 2, shortcut, g));
        }
    }

    register_module("m", m);
}

void A2C2fImpl::check_args(std::vector<arg_complex>& args)
{
    int argsize = args.size();
    if(argsize != 3 && argsize != 5)
    {
        LOG(WARNING) << "A2C2f need 3 args [c2, True/False [4/1/-1]], and not extend type.";
    }
}

torch::Tensor A2C2fImpl::forward(torch::Tensor x)
{
    std::vector<torch::Tensor> y;
    y.emplace_back(cv1->forward(x));

    for(int i = 0; i < m->size(); i++)
    {
        if (auto* seqptr = m[i]->as<A2C2f_2ABlock>())
            y.push_back(seqptr->forward(y.back()));
        else if (auto* c3kptr = m[i]->as<C3k>())
            y.push_back(c3kptr->forward(y.back()));
    }
    at::Tensor y_cat = cv2->forward(torch::cat(y, 1));
//    std::cout << " y " << y.size() << " y_cat: " << y_cat.sizes() << std::endl;
    
    if(a2_ && residual_)
    {
        torch::Tensor ret = (x + this->gamma.view({ -1, gamma.size(0), 1, 1 }) * y_cat);
        return ret;
    }
    return y_cat;
}

torch::Tensor A2C2fImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
//------------------------------ End A2C2f ----------------------------

//============================== Start C2PSA ============================
C2PSAImpl::C2PSAImpl(int c1, int c2, int n, float e)
{
    Init_Modules(c1, c2, n, e);
}

void C2PSAImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    // [-1, 2, C2PSA, [1024]]  c1 == c2
    if (in_channels != std::get<int>(args[0]))
    {
        LOG(ERROR) << "C2PSAImpl c1 == c2 c1: " << in_channels
            << " c2: " << std::get<int>(args[0]);
    }
    check_args(args);
    int argsize = args.size();
    if (argsize == 1)
    {
        //        std::cout << "Init width 3 args.\n";
        Init_Modules(in_channels,
            std::get<int>(args[0]),
            number);
    }
    else {
        std::cout << "Init width 2 args.\n";
        Init_Modules(in_channels,
            std::get<int>(args[0]),
            number,
            std::get<float>(args[1]));
    }
}

void C2PSAImpl::Init_Modules(int c1, int c2, int n, float e)
{
    in_ch = c1;
    out_ch = c2;
    number = n;

    c = int(float(c1) * e);
    cv1 = Conv(c1, 2*c, 1, 1);
    cv2 = Conv(2 * c, c1, 1);
    register_module("cv1", cv1);
    register_module("cv2", cv2);

    int c_64 = static_cast<int>(std::trunc(float(c) / 64.f));
    //std::cout << "c: " << c << " c_64: " << c_64 << std::endl;
    for (int i = 0; i < n; i++)
    {
        m->push_back(*(PSABlock(c, 0.5, c_64)));
    }

    register_module("m", m);
}

void C2PSAImpl::check_args(std::vector<arg_complex>& args)
{
}

torch::Tensor C2PSAImpl::forward(torch::Tensor x)
{
    auto cv1_x = cv1->forward(x).split({ c, c }, 1);
    auto a = cv1_x[0];
    auto b = cv1_x[1];
    b = m->forward(b);
    return cv2->forward(torch::cat({ a, b }, 1));
}

torch::Tensor C2PSAImpl::forward(std::vector<torch::Tensor> x)
{
    LOG(ERROR) << "This module not accept mult-input";
    return x[0];
}
//------------------------------ End C2PSA ----------------------------