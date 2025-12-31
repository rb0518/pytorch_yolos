#include "block.h"

#include "vectorTotuple.h"

//------------------------------ Start DFL ----------------------------
DFLImpl::DFLImpl(int c1)
{
    this->_c1 = c1;
    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, 1, 1).bias(false));
     conv->weight.requires_grad_(false);
     register_module("conv", conv);

    auto x = torch::arange(c1, torch::kFloat);
    conv->weight.data() = x.view({ 1, c1, 1, 1 });

}

torch::Tensor DFLImpl::forward(torch::Tensor x)
{
    // b, _, a = x.shape #batch, channels, anchors
    int b = x.size(0);
    int a = x.size(2);
    return conv->forward(x.view({ b, 4, this->_c1, a }).transpose(2, 1).softmax(1)).view({ b, 4, a });
}
//------------------------------ End DFL ----------------------------

//------------------------------ Start Proto ----------------------------
ProtoImpl::ProtoImpl(int c1, int c_, int c2)
{
    cv1 = Conv(c1, c_, 3);
    cv2 = Conv(c_, c_, 3);
    cv3 = Conv(c_, c2);

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
// 暂时将参数写死，后续修改arg_complex, 能够读取vector<int>, 保留Init_Modules，利用默认参数

SPPImpl::SPPImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    Init_Modules(in_channels, std::get<int>(args[0]));
}

void SPPImpl::Init_Modules(int c1, int c2, std::vector<int> k)
{
    _out_channels = c2;

    int c_ = c1 / 2; //static_cast<int>(std::trunc(float(c1) / 2.0));

    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv(c_ * (k.size() + 1), c2, 1, 1);

    for (int i = 0; i < k.size(); i++)
    {
        int p = k[i] / 2;   //static_cast<int>(std::trunc(k[i] / 2));
        auto options = torch::nn::MaxPool2dOptions(k[i]).stride(1).padding(p);
        //std::cout << "SPP init stride: " << options.stride();
        m->push_back(torch::nn::MaxPool2d(options));
    }

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("m", m);
}

torch::Tensor SPPImpl::forward(torch::Tensor x)
{
    std::vector<torch::Tensor> outputs;
    outputs.emplace_back(cv1->forward(x));

    for (auto& layer : *m)
    {
        outputs.emplace_back(layer->as<torch::nn::MaxPool2d>()->forward(x));   //??? x?
    }

    return cv2->forward(torch::cat(outputs, 1));
}
//------------------------------ End SPP ----------------------------

//============================== SPPFImpl ===========================

SPPFImpl::SPPFImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    auto params = get_defaults();
    vectorvariantToTuple<Conf_Params, arg_complex>(params, args);
    Init_Modules(in_channels, std::get<0>(params), std::get<1>(params));
}

void SPPFImpl::Init_Modules(int c1, int c2, int k)
{
    _out_channels = c2;
    int c_ = c1 / 2;    //static_cast<int>(std::trunc(float(c1) / 2.0));
    int p = k / 2;  // static_cast<int>(std::trunc(k / 2));

    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv(c_ * 4, c2, 1, 1);
    m = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ k, k }).stride(1).padding(p));
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("m", m);
}

torch::Tensor SPPFImpl::forward(torch::Tensor x)
{
    std::vector<torch::Tensor> y;
    y.emplace_back(cv1->forward(x));
    for (int i = 0; i < 3; i++)
        y.emplace_back(m->forward(y.back()));
    return cv2->forward(torch::cat(y, 1));
}
//------------------------------ End SPPF ----------------------------

//============================== Start C2f ============================

C2fImpl::C2fImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    auto params = get_defaults();
    vectorvariantToTuple<Conf_Params, arg_complex>(params, args);
    C2fImpl(in_channels, std::get<0>(params), number, std::get<1>(params),
        std::get<2>(params), std::get<3>(params));
}

C2fImpl::C2fImpl(int c1, int c2, int n, bool shortcut, int g, float e)
{
    this->_out_channels = c2;
    this->_c = int(float(c2) * e);

    cv1 = Conv(c1, 2 * this->_c, 1, 1);
    cv2 = Conv((2 + n) * this->_c, c2, 1);

    register_module("cv1", cv1);
    register_module("cv2", cv2);

    for (int i = 0; i < n; i++) {
        m->push_back(Bottleneck(this->_c, this->_c, shortcut, g, std::make_tuple(3, 3), 1.0f));
    }
    register_module("m", m);
}

torch::Tensor C2fImpl::forward(torch::Tensor x)
{
    auto y = cv1->forward(x).chunk(2, 1);
    for (int i = 0; i < m->size(); i++)
        y.emplace_back(m[i]->as<Bottleneck>()->forward(y.back()));
    
    return cv2->forward(torch::cat(y, 1));
}

torch::Tensor C2fImpl::forward_split(torch::Tensor x)
{
    auto y_split = cv1->forward(x).split({ this->_c, this->_c }, 1);
    std::vector<torch::Tensor> y;
    y.emplace_back(y_split[0]);
    y.emplace_back(y_split[1]);
    for (int i = 0; i < m->size(); i++)
        y.emplace_back(m[i]->as<Bottleneck>()->forward(y.back()));

    return cv2->forward(torch::cat(y, 1));
}
//------------------------------ End C2f ----------------------------

//============================== Start C3k2 ============================

C3k2Impl::C3k2Impl(int in_channels, int number, std::vector<arg_complex>& args)
{
    auto params = get_defaults();
    vectorvariantToTuple<Conf_Params, arg_complex>(params, args);
    Init(in_channels, std::get<0>(params), number,
        std::get<1>(params), std::get<2>(params), std::get<3>(params), std::get<4>(params));
}

C3k2Impl::C3k2Impl(int c1, int c2, int n, bool c3k, float e, int g, bool shortcut)
{
    Init(c1, c2, n, c3k, e, g, shortcut);
}

torch::Tensor C3k2Impl::forward(torch::Tensor x)
{
    auto y = cv1->forward(x).chunk(2, 1);
    for (int i = 0; i < m->size(); i++)
    {
        auto tmp = y[y.size() - 1]; // 取队列最后一个
        if(_b_c3k)
            y.emplace_back(m[i]->as<C3k>()->forward(tmp));
        else
            y.emplace_back(m[i]->as<Bottleneck>()->forward(tmp));
    }
    return cv2->forward(torch::cat(y, 1));
}

torch::Tensor C3k2Impl::forward_split(torch::Tensor x)
{
    auto y = cv1->forward(x).split({ _c, _c }, 1);
    for (int i = 0; i < m->size(); i++)
    {
        auto tmp = y[y.size() - 1]; // 取队列最后一个
        if (_b_c3k)
            y.emplace_back(m[i]->as<C3k>()->forward(tmp));
        else
            y.emplace_back(m[i]->as<Bottleneck>()->forward(tmp));
    }
    return cv2->forward(torch::cat(y, 1));
}
void C3k2Impl::Init(int c1, int c2, int n, bool c3k, float e, int g, bool shortcut)
{
    this->_out_channels = c2;
    _b_c3k = c3k;
    _c = int(float(c2) * e);

    cv1 = Conv(c1, 2 * _c, 1, 1);
    cv2 = Conv((2 + n) * _c, c2, 1);

    register_module("cv1", cv1);
    register_module("cv2", cv2);

    for (int i = 0; i < n; i++) {
        if (_b_c3k)
            m->push_back(C3k(_c, _c, 2, shortcut, g));
        else
            m->push_back(Bottleneck(_c, _c, shortcut, g));
    }
    register_module("m", m);
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

C3Impl::C3Impl(int in_channels, int number, std::vector<arg_complex>& args)
{
    auto params = get_defaults();
    vectorvariantToTuple(params, args);
    C3Impl(in_channels, std::get<0>(params), number, std::get<1>(params),
            std::get<2>(params), std::get<3>(params));
}

C3Impl::C3Impl(int c1, int c2, int n, bool shortcut, int g, float e)
{
    _out_channels = c2;
    int c_ = int(float(c2) * e);
    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv(c1, c_, 1, 1);
    cv3 = Conv(2 * c_, c2, 1);

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    std::vector<std::tuple<int, int>> ks;
    ks.emplace_back(std::make_tuple(1, 1));
    ks.emplace_back(std::make_tuple(3, 3));
    for(int i = 0; i < n; i++)
        m->push_back(Bottleneck(c_, c_, shortcut, g, ks[i % ks.size()], 1.0));
    
    register_module("m", m);
}

torch::Tensor C3Impl::forward(torch::Tensor x)
{
    return cv3->forward(torch::cat({m->forward(cv1->forward(x)), cv2->forward(x)}, 1));
}
//------------------------------ End C3 ----------------------------

//============================== Start C3x ============================

C3xImpl::C3xImpl(int c1, int c2, int n, bool shortcut, int g, float e)
{
    int c_ = int(float(c2) * e);
    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv(c1, c_, 1, 1);
    cv3 = Conv(2 * c_, c2, 1);

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    std::vector<std::tuple<int, int>> ks;
    ks.emplace_back(std::make_tuple(1, 3));
    ks.emplace_back(std::make_tuple(3, 1));
    for (int i = 0; i < n; i++)
        m->push_back(Bottleneck(c_, c_, shortcut, g, ks[i % ks.size()], 1.0));

    register_module("m", m);
}

torch::Tensor C3xImpl::forward(torch::Tensor x)
{
    return cv3->forward(torch::cat({m->forward(cv1->forward(x)), cv2->forward(x)}, 1));
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
    int c_ = int(float(c2) * e);
    auto [k0, k1] = k;
    cv1 = Conv(c1, c_, k0, 1);
    cv2 = Conv(c_, c2, k1, 1, std::nullopt, g);
    register_module("cv1", cv1);
    register_module("cv2", cv2);

    _b_add = (shortcut && c1 == c2);
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x)
{
    if(_b_add)
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


A2C2fImpl::A2C2fImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    set_params(in_channels, number, args);
}

A2C2fImpl::A2C2fImpl(int c1, int c2, int n, bool a2 , int area, bool residual,
        float mlp_ratio, float e, int g, bool shortcut)
{
    Init_Modules(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut);
}

void A2C2fImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args)
{
    // [-1, 4, A2C2f, [512, True, 4/1/-1]] ==> c2 = 512,  n = 4,  a2 = True, area = 4/1/-1
    auto defs = get_defaults();
    vectorvariantToTuple<Conf_Params, arg_complex>(defs, args);

    Init_Modules(in_channels, std::get<0>(defs), number, std::get<1>(defs),
        std::get<2>(defs), std::get<3>(defs), std::get<4>(defs), std::get<5>(defs),
        std::get<6>(defs), std::get<7>(defs));
}

void A2C2fImpl::Init_Modules(int c1, int c2, int n, bool a2 , int area, bool residual, 
        float mlp_ratio, float e, int g, bool shortcut)
{
    _out_channels = c2;
    _a2 = a2;

    int c_ = int(float(c2) * e);
    if(c_ % 32 != 0)
        LOG(ERROR) << "c_ must be multiple of 32." << c_;

    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv((1 + n) * c_, c2, 1);
    register_module("cv1", cv1);
    register_module("cv2", cv2);

    if (a2 && residual)
    {
        gamma = register_parameter("gamma", 0.01 * torch::ones(c2)); // 0.01 or small
        gamma.requires_grad_(true);
    }

    for(int i = 0; i < n ; i++)
    {
        if(a2)
            m->push_back(torch::nn::Sequential(*ABlock(c_, c_/32, mlp_ratio, area),
                    *ABlock(c_, c_ / 32, mlp_ratio, area)));
        else
            m->push_back(C3k(c_, c_, 2, shortcut, g));
    }

    register_module("m", m);
}

torch::Tensor A2C2fImpl::forward(torch::Tensor x)
{
    std::vector<torch::Tensor> y;
    y.emplace_back(cv1->forward(x));

    for(int i = 0; i < m->size(); i++)
    {
        if (this->_a2)
            y.push_back(m[i]->as<torch::nn::Sequential>()->forward(y.back()));
        else
            y.push_back(m[i]->as<C3k>()->forward(y.back()));
    }
    at::Tensor y_cat = cv2->forward(torch::cat(y, 1));
//    std::cout << " y " << y.size() << " y_cat: " << y_cat.sizes() << std::endl;
    
    if(this->named_parameters().find("gamma")!=nullptr)
        return x + this->gamma.view({ 1, -1, 1, 1 }) * y_cat;

    return y_cat;
}

//------------------------------ End A2C2f ----------------------------

//============================== Start C2PSA ============================

C2PSAImpl::C2PSAImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    set_params(in_channels, number, args);
}

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
    int args_size = args.size();
    if (args_size == 1)
    {
        //        std::cout << "Init width 3 args.\n";
        Init_Modules(in_channels,
            std::get<int>(args[0]),
            number);
    }
    else if (args_size == 2)
    {
        //std::cout << "Init width 2 args.\n";
        Init_Modules(in_channels,
            std::get<int>(args[0]),
            number,
            std::get<float>(args[1]));
    }
    else
        LOG(ERROR) << "C2PSAImpl number of args wrong(1 or 2): " << args_size;
}

void C2PSAImpl::Init_Modules(int c1, int c2, int n, float e)
{
    _out_channels = c2;
    _c = int(float(c1) * e);

    cv1 = Conv(c1, 2*_c, 1, 1);
    cv2 = Conv(2 * _c, c1, 1);

    register_module("cv1", cv1);
    register_module("cv2", cv2);

    int c_64 = _c / 64; // static_cast<int>(std::trunc(float(c) / 64.f));
    //std::cout << "c: " << c << " c_64: " << c_64 << std::endl;
    for (int i = 0; i < n; i++)
        m->push_back(*(PSABlock(_c, 0.5, c_64)));

    register_module("m", m);
}

torch::Tensor C2PSAImpl::forward(torch::Tensor x)
{
    auto cv1_x = cv1->forward(x).split({ _c, _c }, 1);
    auto a = cv1_x[0];
    auto b = cv1_x[1];
    b = m->forward(b);
    return cv2->forward(torch::cat({ a, b }, 1));
}

//------------------------------ End C2PSAImpl ----------------------------

//============================== Start HyperACEImpl ============================

HyperACEImpl::HyperACEImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    auto params = get_defaults();
    vectorvariantToTuple<Conf_Params, arg_complex>(params, args);

    Init(in_channels, std::get<0>(params), number,
        std::get<1>(params),
        std::get<2>(params),
        std::get<3>(params),
        std::get<4>(params),
        std::get<5>(params),
        std::get<6>(params));
}

HyperACEImpl::HyperACEImpl(int c1, int c2, int n, int num_hyperedges, bool dsc3k, bool shortcut, float e1, float e2, std::string context, bool channel_adjust)
{
    Init(c1, c2, n, num_hyperedges, dsc3k, shortcut, e1, e2, context, channel_adjust);
};

torch::Tensor HyperACEImpl::forward(std::vector<torch::Tensor> x)
{
    auto x_fuse = fuse->forward(x);
    auto y = cv1->forward(x_fuse).chunk(3, 1);
    auto out1 = branch1->forward(y[1]);
    auto out2 = branch2->forward(y[1]);

    for (int i = 0; i < m->size(); i++)
    {
        if (_b_dsc3k)
            y.push_back(m[i]->as<DSC3k>()->forward(y.back()));
        else
            y.push_back(m[i]->as<DSBottleneck>()->forward(y.back()));

    }
    y[1] = out1;
    y.push_back(out2);
    
    return cv2->forward(torch::cat(y, 1));       
}
void HyperACEImpl::Init(int c1, int c2, int n, int num_hyperedges, bool dsc3k, bool shortcut, float e1, float e2, std::string context, bool channel_adjust)
{
    _out_channels = c2;
    _b_dsc3k = dsc3k;

    int c = int(float(c2) * e1);
    this->cv1 = Conv(c1, 3 * c, 1, 1);
    this->cv2 = Conv((4 + n) * c, c2, 1);

    for (int i = 0; i < n; i++)
    {
        if (dsc3k)
        {
            /*
            DSC3kImpl(int c1, int c2, int n = 1, bool shortcut = true, int g = 1,
                float e = 0.5, int k1 = 3, int k2 = 5, int d2 = 1)            */
            m->push_back(DSC3k(c, c, 2, shortcut, 1, 0.5, 3, 7));
        }
        else {
            m->push_back(DSBottleneck(c, c, shortcut));
        }
    }

    fuse = FuseModule(c1, channel_adjust);
    branch1 = C3AH(c, c, e2, num_hyperedges, context);
    branch2 = C3AH(c, c, e2, num_hyperedges, context);

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("m", m);
    register_module("fuse", fuse);
    register_module("branch1", branch1);
    register_module("branch2", branch2);
}
//------------------------------ End HyperACEImpl ----------------------------

//=============================== Start DownsampleConv ========================

DownsampleConvImpl::DownsampleConvImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    int args_size = args.size();
    if (args_size == 1)
        Init(in_channels, std::get<bool>(args[0]));
    else
        LOG(WARNING) << "default args is empty() == true";
}

DownsampleConvImpl::DownsampleConvImpl(int in_channels, bool channel_adjust)
{
    Init(in_channels, channel_adjust);
}

torch::Tensor DownsampleConvImpl::forward(torch::Tensor x)
{
    if (false == _b_channel_adjust)
        return downsample->forward(x);
    return cv_channel_adjust->forward(downsample->forward(x));
}
void DownsampleConvImpl::Init(int in_channels, bool channel_adjust)
{
    _out_channels = channel_adjust ? in_channels * 2 : in_channels;
    _b_channel_adjust = channel_adjust;
    downsample = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ 2, 2 }));
    register_module("downsample", downsample);

    if (channel_adjust)
    {
        cv_channel_adjust = Conv(in_channels, in_channels * 2, 1);
        register_module("channel_adjust", cv_channel_adjust);
    }
}
//-------------------------------- End DownsampleConv -----------------------------

//=============================== Start DSC3k2 ========================

DSC3k2Impl::DSC3k2Impl(int in_channels, int number, std::vector<arg_complex>& args)
{
    vectorvariantToTuple<Conf_Params, arg_complex>(def_params, args);
    Init(in_channels, std::get<0>(def_params), number, std::get<1>(def_params), 
            std::get<2>(def_params), std::get<3>(def_params),
            std::get<4>(def_params), std::get<5>(def_params),
            std::get<6>(def_params), std::get<7>(def_params));
}


DSC3k2Impl::DSC3k2Impl(int c1, int c2, int n, bool dsc3k, float e, int g, bool shortcut, int k1, int k2, int d2)
{
    Init(c1, c2, n, dsc3k, e, g, shortcut, k1, k2, d2);
}

torch::Tensor DSC3k2Impl::forward(torch::Tensor x)
{
    auto y = cv1->forward(x).chunk(2, 1);
    for (int i = 0; i < m->size(); i++)
    {
        if (_b_dsc3k)
            y.push_back(m[i]->as<DSC3k>()->forward(y.back()));
        else
            y.push_back(m[i]->as<DSBottleneck>()->forward(y.back()));
    }
    return cv2->forward(torch::cat(y, 1));
}
void DSC3k2Impl::Init(int c1, int c2, int n, bool dsc3k, float e, int g, bool shortcut, int k1, int k2, int d2)
{
    _out_channels = c2;
    _b_dsc3k = dsc3k;

    int c = int(float(c2) * e);
    cv1 = Conv(c1, 2 * c, 1, 1);
    cv2 = Conv((2 + n) * c, c2, 1);

    for (int i = 0; i < n; i++)
    {
        if (dsc3k)
        {
            /*
            DSC3kImpl(int c1, int c2, int n = 1, bool shortcut = true, int g = 1,
                float e = 0.5, int k1 = 3, int k2 = 5, int d2 = 1)
            */
            m->push_back(DSC3k(c, c, 2, shortcut, g, 1.0, k1, k2, d2));
        }
        else {
            m->push_back(DSBottleneck(c, c, shortcut, 1.0, k1, k2, d2));
        }
    }

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("m", m);
}
//-------------------------------- End DSC3k2 -----------------------------

BottleneckCSPImpl::BottleneckCSPImpl(int c1, int c2, int n, bool shortcut_, int g, float e)
{
    int c_ = int(float(c2) * e);
    cv1 = Conv(c1, c_, 1, 1);
    cv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c_, 1).stride(1).bias(false));
    cv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(c_, c_, 1).stride(1).bias(false));
    cv4 = Conv(2 * c_, c2, 1, 1);
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(2 * c_));
    act = torch::nn::SiLU();
    for (int i = 0; i < n; i++)
        m->push_back(*(Bottleneck(c_, c_, shortcut_, g, std::make_tuple(3, 3), 1.0)));

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    register_module("cv4", cv4);
    register_module("bn", bn);
    register_module("act", act);
    register_module("m", m);
}

torch::Tensor BottleneckCSPImpl::forward(torch::Tensor x)
{
    torch::Tensor y1 = cv3->forward(m->forward(cv1->forward(x)));
    torch::Tensor y2 = cv2->forward(x);
    return cv4->forward(act->forward(bn->forward(torch::cat({ y1, y2 }, 1))));
}
//-------------------------------- End BottleneckCSPImpl -----------------------------

AAttnImpl::AAttnImpl(int dim, int num_heads, int area)
    : _area(area), _num_heads(num_heads)
{
    this->_head_dim = dim / num_heads;
    int all_head_dim = _head_dim * num_heads;

    qkv_cv = Conv(dim, all_head_dim * 3, 1, 1, std::nullopt, 1, 1, "False");
    proj_cv = Conv(all_head_dim, dim, 1, 1, std::nullopt, 1, 1, "False");
    pe_cv = Conv(all_head_dim, dim, 7, 1, 3, dim, 1, "False");
    register_module("qkv", qkv_cv);
    register_module("proj", proj_cv);
    register_module("pe", pe_cv);
}

torch::Tensor AAttnImpl::forward(torch::Tensor x)
{
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int N = H * W;

    auto qkv = qkv_cv->forward(x).flatten(2).transpose(1, 2);
    if (_area > 1)
    {
        auto N_area = static_cast<int>(std::trunc(float(N) / float(_area)));
        qkv = qkv.reshape({ B * _area, N / _area, C * 3 });
        B = qkv.size(0);
        N = qkv.size(1);
    }

    auto qkv_split = qkv.view({ B, N, _num_heads, _head_dim * 3 }).permute({ 0, 2, 3, 1 }).split(
        { _head_dim, _head_dim, _head_dim }, 2);
    torch::Tensor q = qkv_split[0];
    torch::Tensor k = qkv_split[1];
    torch::Tensor v = qkv_split[2];

    auto attn = (q.transpose(-2, -1).matmul(k)) * std::pow(_head_dim, -0.5);
    attn = attn.softmax(-1);
    auto x_ret = v.matmul(attn.transpose(-2, -1));
    x_ret = x_ret.permute({ 0, 3, 1, 2 });
    v = v.permute({ 0, 3, 1, 2 });

    if (_area > 1)
    {
        x_ret = x_ret.reshape({ B / _area, N * _area, C });
        v = v.reshape({ B / _area, N * _area, C });
        B = x_ret.size(0);
        N = x_ret.size(1);
    }

    x_ret = x_ret.reshape({ B, H, W, C }).permute({ 0, 3, 1, 2 }).contiguous();
    v = v.reshape({ B, H, W, C }).permute({ 0, 3, 1, 2 }).contiguous();
    x_ret = x_ret + pe_cv->forward(v);
    return proj_cv->forward(x_ret);
}

//-------------------------------- End AAttnImpl -----------------------------

ABlockImpl::ABlockImpl(int dim, int num_heads, float mlp_ratio, int area)
{
    //  std::cout << "ABlock input dim: " << dim << " num_heads: " << num_heads
    //      << " ratio: " << mlp_ratio << " area: " << area << "\n";

    attn = AAttn(dim, num_heads, area);
    int mlp_hidden_dim = int(float(dim) * mlp_ratio);

    register_module("attn", attn);
    mlp = torch::nn::Sequential(
            Conv(dim, mlp_hidden_dim, 1),
            Conv(mlp_hidden_dim, dim, 1, 1, std::nullopt, 1, 1, "False")
        );
    register_module("mlp", mlp);

    for (auto& sub_model : this->modules(false))
    {
        this->init_weights(*sub_model);
    }
}

torch::Tensor ABlockImpl::forward(torch::Tensor x)
{
    auto y = x + attn->forward(x);
    return y + mlp->forward(y);
}

/*
    """
    Initialize weights using a truncated normal distribution.

    Args:
        m (nn.Module): Module to initialize.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
*/

torch::Tensor _no_grad_trunc_normal_(torch::Tensor tensor, float mean, float std,
    float a, float b, std::optional<at::Generator> generator = std::nullopt)
{
    auto norm_cdf = [](float x) {
        return (1.0f + std::erff(x / std::sqrt(2.0f))) / 2.0f;
        };

    auto l = norm_cdf((a - mean) / std);
    auto u = norm_cdf((b - mean) / std);

    tensor.uniform_(2 * double(l) - 1.0, 2 * double(u) - 1, generator);
    tensor.erfinv_();

    tensor.mul_(double(std) * std::sqrt(2.0));
    tensor.add_(double(mean));

    tensor.clamp_(a, b);

    return tensor;
}


/*
* past from pytorch, libtorch torch::nn::init namespace not found trunc_normal_ function
*     r"""Fill the input Tensor with values drawn from a truncated normal distribution.
*/
torch::Tensor _trunc_normal_(torch::Tensor tensor, float mean = 0.0f,
    float std = 1.0f, float a = -2.0f, float b = 2.0f, 
    std::optional<at::Generator> generator = std::nullopt)
{
    return _no_grad_trunc_normal_(tensor, mean, std, a, b, generator);
}


void ABlockImpl::init_weights(torch::nn::Module& module)
{
    if (auto* conv = module.as<torch::nn::Conv2d>()) 
    {
        _trunc_normal_(conv->weight, 0.0f, 0.02f);
        if (conv->bias.defined())
        {
            torch::nn::init::constant_(conv->bias, 0);
        }
        // else
        //     std::cout << "conv bias not defined.\n";

    } //原代码只初始化了nn.Conv2d
    // else if (auto* bn = module.as<torch::nn::BatchNorm2d>()) {
    //     torch::nn::init::ones_(bn->weight);
    //     torch::nn::init::zeros_(bn->bias);
    // }
}
//-------------------------------- End ABlockImpl -----------------------------

AttentionImpl::AttentionImpl(int dim, int num_heads, float attn_ratio)
{
    this->_num_heads = num_heads;
    this->_head_dim = dim / num_heads;  //static_cast<int>(std::trunc(float(dim) / float(num_heads)));
    this->_key_dim = int(this->_head_dim * attn_ratio);
    this->_scale = std::pow(this->_key_dim, -0.5);

    int nh_kd = this->_key_dim * num_heads;
    int h = dim + nh_kd * 2;

    cv_qkv = Conv(dim, h, 1, 1, std::nullopt, 1, 1, "False");
    cv_proj = Conv(dim, dim, 1, 1, std::nullopt, 1, 1, "False");
    cv_pe = Conv(dim, dim, 3, 1, std::nullopt, dim, 1, "False");
    register_module("qkv", cv_qkv);
    register_module("proj", cv_proj);
    register_module("pe", cv_pe);
}

torch::Tensor AttentionImpl::forward(torch::Tensor x)
{
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int N = H * W;

    auto qkv = cv_qkv->forward(x);
    auto qkv_split = qkv.view({ B, this->_num_heads, this->_key_dim * 2 + this->_head_dim, N }).split(
        { this->_key_dim, this->_key_dim, this->_head_dim }, 2);
    auto q = qkv_split[0];
    auto k = qkv_split[1];
    auto v = qkv_split[2];

    auto attn = (q.transpose(-2, -1).matmul(k)) * this->_scale;
    attn = attn.softmax(-1);
    auto y = (v.matmul(attn.transpose(-2, -1))).view({ B, C, H, W }) +
        cv_pe->forward(v.reshape({ B, C, H, W }));
    y = cv_proj->forward(y);

    return y;
}
//-------------------------------- End AttentionImpl -----------------------------

PSABlockImpl::PSABlockImpl(int c, float attn_ratio, int num_heads, bool short_cut)

{
    this->_b_add = short_cut;

    attn = Attention(c, num_heads, attn_ratio);

    ffn = torch::nn::Sequential(
        Conv(c, c * 2, 1),
        Conv(c * 2, c, 1, 1, std::nullopt, 1, 1, "False")
        );
    register_module("attn", attn);
    register_module("ffn", ffn);
}

torch::Tensor PSABlockImpl::forward(torch::Tensor x)
{
    torch::Tensor y;
    if (_b_add)
    {
        y = x + attn->forward(x);
        y = y + ffn->forward(y);
    }
    else
    {
        y = attn->forward(x);
        y = ffn->forward(y);
    }

    return y;
}
//-------------------------------- End PSABlockImpl -----------------------------

DSBottleneckImpl::DSBottleneckImpl(int c1, int c2, bool shortcut, float e, int k1, int k2, int d2)
{
    int c_ = int(float(c2) * e);
    cv1 = DSConv(c1, c_, k1, 1, std::nullopt, 1);
    cv2 = DSConv(c_, c2, k2, 1, std::nullopt, d2);
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    this->_b_add = shortcut && c1 == c2;
}

torch::Tensor DSBottleneckImpl::forward(torch::Tensor x) {
    auto y = cv2->forward(cv1->forward(x));
    if (_b_add)
        return x + y;
    return y;
}

//-------------------------------- End DSBottleneckImpl -----------------------------

DSC3kImpl::DSC3kImpl(int c1, int c2, int n, bool shortcut, int g, float e, int k1, int k2, int d2)
{

    int c_ = int(float(c2) * e);
    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv(c1, c_, 1, 1);
    cv3 = Conv(2 * c_, c2, 1);

    for (int i = 0; i < n; i++)
    {
        /*
        DSBottleneckImpl(int c1, int c2, bool shortcut = true,
            float e = 0.5f, int k1 = 3, int k2 = 5, int d2 = 1)
        */

        m->push_back(*(DSBottleneck(
            c_, c_, shortcut, 1.0, k1, k2, d2
        )));
    }

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    register_module("m", m);
}

torch::Tensor DSC3kImpl::forward(torch::Tensor x) {
    return cv3->forward(torch::cat({
        m->forward(cv1->forward(x)),
        cv2->forward(x)
        }, 1));
}

//-------------------------------- End DSC3kImpl -----------------------------

FuseModuleImpl::FuseModuleImpl(int c_in, bool channel_adjust)
{
    this->downsample = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2));
    this->upsample = torch::nn::Upsample(
        torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2.0, 2.0})
        .mode(torch::kNearest));
    if (channel_adjust)
        conv_out = Conv(4 * c_in, c_in, 1);
    else
        conv_out = Conv(3 * c_in, c_in, 1);

    register_module("downsample", downsample);
    register_module("upsample", upsample);
    register_module("conv_out", conv_out);
}

torch::Tensor FuseModuleImpl::forward(std::vector<torch::Tensor> x)
{
    auto x1_ds = downsample->forward(x[0]);
    auto x3_up = upsample->forward(x[2]);
    auto x_cat = torch::cat({ x1_ds, x[1], x3_up }, 1);
    auto out = conv_out->forward(x_cat);
    return out;
}

//-------------------------------- End FuseModuleImpl -----------------------------

AdaHyperedgeGenImpl::AdaHyperedgeGenImpl(int node_dim, int num_hyperedges, int num_heads, float dropout, std::string context)
{
    this->_num_heads = num_heads;
    this->_num_hyperedges = num_hyperedges;
    this->_context = context;
    this->_head_dim = node_dim / num_heads;  // static_cast<int>(std::trunc(float(node_dim) / float(num_heads)));

    prototype_base = register_parameter("prototype_base",
        torch::empty({ num_hyperedges, node_dim }));
    torch::nn::init::xavier_uniform_(prototype_base);

    if (context == "mean" || context == "max")
        context_net = torch::nn::Linear(torch::nn::LinearOptions(
            node_dim, num_hyperedges * node_dim));
    else if (context == "both")
        context_net = torch::nn::Linear(torch::nn::LinearOptions(
            2 * node_dim, num_hyperedges * node_dim));
    else
        LOG(ERROR) << "context wrong: " << context;

    pre_head_proj = torch::nn::Linear(torch::nn::LinearOptions(
        node_dim, node_dim));

    // python __init__(self, p: float = 0.5, inplace: bool = False) -> None:
    dropout_net = torch::nn::Dropout(torch::nn::DropoutOptions(double(dropout)));

    this->_scaling = std::sqrt(this->_head_dim);

    register_module("context_net", context_net);
    register_module("pre_head_proj", pre_head_proj);
    register_module("dropout", dropout_net);
}

torch::Tensor AdaHyperedgeGenImpl::forward(torch::Tensor x)
{
    int B = x.size(0);
    int N = x.size(1);
    int D = x.size(2);
    torch::Tensor context_cat;

    if (_context == "mean")
    {
        context_cat = x.mean(1);
    }
    else if (_context == "max")
    {
        torch::Tensor m_idx;
        std::tie(context_cat, m_idx) = x.max(1);
    }
    else
    {
        auto avg_context = x.mean(1);
        auto [max_context, max_idx] = x.max(1);
        context_cat = torch::cat({ avg_context, max_context }, -1);
    }

    auto prototype_offset = context_net->forward(context_cat)
        .view({ B, _num_hyperedges, D });
    auto prototypes = prototype_base.unsqueeze(0) + prototype_offset;

    auto x_proj = pre_head_proj->forward(x);

    auto x_heads = x_proj.view({ B, N, _num_heads, _head_dim }).transpose(1, 2);

    auto proto_heads = prototypes.view({ B, _num_hyperedges, _num_heads,
            _head_dim }).permute({ 0, 2, 1, 3 });

    auto x_heads_flat = x_heads.reshape({ B * _num_heads, N, _head_dim });

    auto proto_heads_flat = proto_heads.reshape({ B * _num_heads, _num_hyperedges,
        _head_dim }).transpose(1, 2);

    torch::Tensor logits = torch::bmm(x_heads_flat, proto_heads_flat) / this->_scaling;
    logits = logits.view({ B, _num_heads, N, _num_hyperedges }).mean(1);

    logits = dropout_net->forward(logits);

    return torch::nn::functional::softmax(logits, torch::nn::functional::SoftmaxFuncOptions(1));
}

//-------------------------------- End AdaHyperedgeGenImpl -----------------------------

AdaHGConvImpl::AdaHGConvImpl(int embed_dim, int num_hyperedge, int num_heads, float dropout, std::string context)
{
    edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedge, num_heads,
        dropout, context);
    edge_proj = torch::nn::Sequential(
        torch::nn::Linear(embed_dim, embed_dim),
        torch::nn::GELU()
    );

    node_proj = torch::nn::Sequential(
        torch::nn::Linear(embed_dim, embed_dim),
        torch::nn::GELU()
    );

    register_module("edge_generator", edge_generator);
    register_module("edge_proj", edge_proj);
    register_module("note_proj", node_proj);
}

torch::Tensor AdaHGConvImpl::forward(torch::Tensor x)
{
    auto A = edge_generator->forward(x);

    auto He = torch::bmm(A.transpose(1, 2), x);

    He = edge_proj->forward(He);
    auto X_new = torch::bmm(A, He);

    X_new = node_proj->forward(X_new);
    return X_new + x;
}
//-------------------------------- End AdaHGConvImpl -----------------------------

AdaHGComputationImpl::AdaHGComputationImpl(int embed_dim, int num_hyperedges, int num_heads, float dropout, std::string context)
{
    hgnn = AdaHGConv(embed_dim, num_hyperedges, num_heads,
        dropout, context);
    register_module("hgnn", hgnn);
}

torch::Tensor AdaHGComputationImpl::forward(torch::Tensor x) {
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    auto tokens = x.flatten(2).transpose(1, 2);
    tokens = hgnn->forward(tokens);

    auto x_out = tokens.transpose(1, 2).view({ B, C, H, W });
    return x_out;
}
//-------------------------------- End AdaHGComputationImpl -----------------------------

C3AHImpl::C3AHImpl(int c1, int c2, float e, int num_hyperedges, std::string context)
{
    int c_ = int(float(c2) * e);
    if ((c_ % 16) != 0)
        LOG(ERROR) << "C3AH c_ not multiple 16";

    int num_heads = c_ / 16;
    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv(c1, c_, 1, 1);

    m_adahg = AdaHGComputation(c_, num_hyperedges,
        num_heads, 0.1, context);

    cv3 = Conv(2 * c_, c2, 1);

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("m", m_adahg);
    register_module("cv3", cv3);
}

torch::Tensor C3AHImpl::forward(torch::Tensor x)
{
    return cv3->forward(torch::cat({
        m_adahg->forward(cv1->forward(x)),
            cv2->forward(x) }, 1));
}

//-------------------------------- End C3AHImpl -----------------------------

FullPAD_TunnelImpl::FullPAD_TunnelImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    Init(in_channels);
}

FullPAD_TunnelImpl::FullPAD_TunnelImpl(int in_channels)
{
    Init(in_channels);
}


torch::Tensor FullPAD_TunnelImpl::forward(std::vector<torch::Tensor> x) {
    return x[0] + gate * x[1];
}
void FullPAD_TunnelImpl::Init(int in_channels)
{
    _out_channels = in_channels;
    this->gate = register_parameter("gate", torch::tensor(0.0));
}
//-------------------------------- End FullPAD_TunnelImpl -----------------------------

//-------------------------------- Start HGStemImpl -----------------------------

HGStemImpl::HGStemImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    // args中数据顺序 [cm, c2]
    Init(in_channels, std::get<int>(args[1]), std::get<int>(args[0]));
}

HGStemImpl::HGStemImpl(int c1, int cm, int c2)
{
    Init(c1, c2, cm);
}

torch::Tensor HGStemImpl::forward(torch::Tensor x)
{
    x = stem1->forward(x);
    auto pad_op = torch::nn::functional::PadFuncOptions({ 0, 1, 0, 1 });
    x = torch::nn::functional::pad(x, pad_op);
    auto x2 = stem2a->forward(x);
    x2 = torch::nn::functional::pad(x2, pad_op);
    x2 = stem2b->forward(x2);
    auto x1 = pool->forward(x);
    x = torch::cat({ x1, x2 }, 1);
    x = stem3->forward(x);
    x = stem4->forward(x);
    return x;
}

void HGStemImpl::Init(int c1, int c2, int cm)
{
    _out_channels = c2;

    stem1 = Conv(c1, cm, 3, 2, std::nullopt, 1, 1, "ReLU");
    stem2a = Conv(cm, cm / 2, 2, 1, 0, 1, 1, "ReLU");
    stem2b = Conv(cm / 2, cm, 2, 1, 0, 1, 1, "ReLU");
    stem3 = Conv(cm * 2, cm, 3, 2, std::nullopt, 1, 1, "ReLU");
    stem4 = Conv(cm, c2, 1, 1, std::nullopt, 1, 1, "ReLU");
    auto op = torch::nn::MaxPool2dOptions(2).stride(1).padding(0).ceil_mode(false);
    pool = torch::nn::MaxPool2d(op);

    register_module("stem1", stem1);
    register_module("stem2a", stem2a);
    register_module("stem2b", stem2b);
    register_module("stem3", stem3);
    register_module("stem4", stem4);
    register_module("pool", pool);
}

//-------------------------------- End HGStemImpl -----------------------------

//-------------------------------- Start HGBlockImpl -----------------------------

HGBlockImpl::HGBlockImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    Conf_Params def_params = get_defaults();
    vectorvariantToTuple<Conf_Params, arg_complex>(def_params, args);
    Init(in_channels, std::get<0>(def_params), std::get<1>(def_params),
        std::get<2>(def_params), number,
        std::get<3>(def_params),
        std::get<4>(def_params), std::get<5>(def_params));
}

torch::Tensor HGBlockImpl::forward(torch::Tensor x)
{
    std::vector<torch::Tensor> vc_y = { x };
    for (int i = 0; i < m->size(); i++)
        vc_y.push_back(m[i]->as<torch::nn::AnyModule>()->forward(vc_y.back()));

    auto y = ec->forward(sc->forward(torch::cat(vc_y, 1)));
    if (m_add)
        return y + x;
    return y;
}

void HGBlockImpl::Init(int c1, int cm, int c2, int k, int n, bool lightconv, bool shortcut, std::string act)
{
    _out_channels = c2;
    if (lightconv)
    {
        for (int i = 0; i < n; i++)
            m->push_back(LightConv((i==0? c1 : cm), cm, k, act));
    }
    else
    {
        for (int i = 0; i < n; i++)
            m->push_back(Conv((i == 0 ? c1 : cm), cm, k, 1, std::nullopt, 1, 1, act));
    }

    sc = Conv(c1 + n * cm, c2 / 2, 1, 1, std::nullopt, 1, 1, act);
    ec = Conv(c2 / 2, c2, 1, 1, std::nullopt, 1, 1, act);
    register_module("m", m);
    register_module("sc", sc);
    register_module("ec", ec);
    m_add = shortcut && c1 == c2;
}

//-------------------------------- End HGBlockImpl -----------------------------

//-------------------------------- Start AIFIImpl -----------------------------

AIFIImpl::AIFIImpl(int in_channels, int number, std::vector<arg_complex>& args)
{
    Conf_Params def_params = get_defaults();
    vectorvariantToTuple<Conf_Params, arg_complex>(def_params, args);
    
    Init(in_channels, std::get<0>(def_params), std::get<1>(def_params),
        std::get<2>(def_params), std::get<3>(def_params));
}

torch::Tensor AIFIImpl::forward(torch::Tensor x)
{
    int c = x.size(1);
    int h = x.size(2);
    int w = x.size(3);
    auto pos_embed = build_2d_sincos_position_embedding(x);

    auto rets = forward_super(x.flatten(2).permute({ 0, 2, 1 }), std::nullopt, std::nullopt, pos_embed);
    return rets.permute({ 0, 2, 1 }).view({ -1,  c, h, w }).contiguous();
}

torch::Tensor AIFIImpl::forward_super(torch::Tensor src, std::optional<torch::Tensor> src_mask, 
    std::optional<torch::Tensor> src_key_padding_mask, std::optional<torch::Tensor> pos)
{
    if (m_normalize_before)
        return forward_pre(src, src_mask, src_key_padding_mask, pos);
    return forward_post(src, src_mask, src_key_padding_mask, pos);
}

torch::Tensor AIFIImpl::build_2d_sincos_position_embedding(torch::Tensor x)
{
    int c = x.size(1);
    int h = x.size(2);
    int w = x.size(3);
    float temperature = 10000.f;

    auto grid_w = torch::arange(w, torch::kFloat32);
    auto grid_h = torch::arange(h, torch::kFloat32);
    auto grid_wh = torch::meshgrid({ grid_w, grid_h }, "ij");
    grid_w = grid_wh[0];
    grid_h = grid_wh[1];

    auto pos_dim = c / 4;
    auto omega = torch::arange(pos_dim, torch::kFloat32) / pos_dim;
    omega = 1.0f / torch::pow(temperature, omega);

    auto out_w = torch::matmul(grid_w.flatten().unsqueeze(-1), omega.unsqueeze(0));
    auto out_h = torch::matmul(grid_h.flatten().unsqueeze(-1), omega.unsqueeze(0));

    auto rets = torch::cat({ torch::sin(out_w), torch::cos(out_w),
        torch::sin(out_h), torch::cos(out_h) }, 1).unsqueeze(0);
    rets = rets.to(x.dtype()).to(x.device());
    return rets;
}

torch::Tensor AIFIImpl::with_pos_embed(torch::Tensor tensor, std::optional<torch::Tensor> pos)
{
    if (pos.has_value())
        return tensor + pos.value();
    return tensor;
}

torch::Tensor AIFIImpl::forward_post(torch::Tensor src, std::optional<torch::Tensor> src_mask, std::optional<torch::Tensor> src_key_padding_mask, std::optional<torch::Tensor> pos)
{
    torch::Tensor q = with_pos_embed(src, pos);
    torch::Tensor k = q;

    auto src2 = std::get<0>(m_ma->forward(q, k, src,
        src_key_padding_mask.value_or(torch::Tensor()),
        true,
        src_mask.value_or(torch::Tensor()),
        true));
    src = src + m_dropout1->forward(src2);
    src = m_norm1->forward(src);
    src2 = m_fc2->forward(m_dropout->forward(m_act->forward(m_fc1->forward(src))));
    src = src + m_dropout2->forward(src2);
    return m_norm2->forward(src);
}

torch::Tensor AIFIImpl::forward_pre(torch::Tensor src, std::optional<torch::Tensor> src_mask, std::optional<torch::Tensor> src_key_padding_mask, std::optional<torch::Tensor> pos)
{
    auto src2 = m_norm1->forward(src);
    auto q = with_pos_embed(src2, pos);
    auto k = q;
    src2 = std::get<0>(m_ma->forward(q, k, src2, 
        src_key_padding_mask.value_or(torch::Tensor()),
        true,
        src_mask.value_or(torch::Tensor()),
        true));
    src = src + m_dropout1->forward(src2);
    src2 = m_norm2->forward(src);
    src2 = m_fc2->forward(m_dropout->forward(m_act->forward(m_fc1->forward(src2))));
    return src + m_dropout2->forward(src2);
}

void AIFIImpl::Init(int c1, int cm, int num_heads, float dropout, bool normalize_before)
{
    auto ma_ops = torch::nn::MultiheadAttentionOptions(c1, num_heads).dropout(dropout)/*.batch_first(true)*/;   // libtorch C++中没有batch_first args defined.
    m_ma = torch::nn::MultiheadAttention(ma_ops);

    m_fc1 = torch::nn::Linear(c1, cm);
    m_fc2 = torch::nn::Linear(cm, c1);

    m_norm1 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({int64_t(c1)}));
    m_norm2 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({int64_t(c1)}));

    m_dropout = torch::nn::Dropout(dropout);
    m_dropout1 = torch::nn::Dropout(dropout);
    m_dropout2 = torch::nn::Dropout(dropout);

    m_act = torch::nn::GELU();

    register_module("ma", m_ma);
    register_module("fc1", m_fc1);
    register_module("fc2", m_fc2);

    register_module("norm1", m_norm1);
    register_module("norm2", m_norm2);

    register_module("dropout", m_dropout);
    register_module("dropout1", m_dropout1);
    register_module("dropout2", m_dropout2);

    register_module("act", m_act);

    m_normalize_before = normalize_before;
}


//-------------------------------- End AIFIImpl -----------------------------

RepC3Impl::RepC3Impl(int in_channels, int number, std::vector<arg_complex>& args)
{
    Init(in_channels, std::get<int>(args[0]), number);
}

torch::Tensor RepC3Impl::forward(torch::Tensor x)
{
    return cv3.forward(m->forward(cv1->forward(x)) + cv2->forward(x));
}

void RepC3Impl::Init(int c1, int c2, int n, float e)
{
    _out_channels = c2;
    auto c_ = int(c2 * e);

    cv1 = Conv(c1, c_, 1, 1);
    cv2 = Conv(c1, c_, 1, 1);
    for (int i = 0; i < n; i++)
    {
        m->push_back(*(RepConv(c_, c_)));
    }
    if (c_ = c2)
        cv3 = std::make_shared<ConvImpl>(c_, c2, 1, 1);
    else
        cv3 = std::make_shared<torch::nn::IdentityImpl>();

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3.ptr());
    register_module("m", m);
}
