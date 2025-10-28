#pragma once
/*
    对应着新版本中nn/conv.py文件中的定义
    包含如下的modules
    "Conv",                 [√]     
    "Conv2",                [√]
    "LightConv",            [√]
    "DWConv",               [√]
    "DWConvTranspose2d",    [√]
    "ConvTranspose",        [√]     未见配置文件中出现，可以继承自torch::nn::Module     
    "Focus",                [√]
    "GhostConv",            [√]
    "ChannelAttention",     [√]
    "SpatialAttention",     [√]
    "CBAM",                 [√]
    "Concat",               [√]
    "RepConv",                      暂时只有rtdetr.yaml中RepC3和RepBottleneck会用到
*/


#include <BaseModel.h>
#include <torch/torch.h>


/*
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
*/
class ConvImpl : public BaseModule
{
    // def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
public:
    ConvImpl(){};
    ~ConvImpl(){};
    explicit ConvImpl(int c1, int c2, int k = 1, int s = 1, std::optional<int> p = std::nullopt, int g = 1, int d = 1, std::string act = "True");
    
    void set_params(int inchannels, int number, std::vector<arg_complex>& args);
    // 简化下转参数的适应性，后续再修改
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override {return out_ch;} ;
    void check_args(std::vector<arg_complex>& args);

    void fuse_conv_and_bn(bool fused_ = true);
public:
    int in_ch;
    int out_ch;
    int k_size;
    int stride;
    int padding;

    bool fused = false;
    bool bfirst = true;
    
    void Init_Modules(int c1, int c2, int k = 1, int s = 1, std::optional<int> p = std::nullopt , int g = 1, int d = 1, std::string act = "True" );
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };

    torch::nn::AnyModule _act;

    torch::nn::Conv2d fusedconv{ nullptr };
};
TORCH_MODULE(Conv);

/*
    """Simplified RepConv module with Conv fusing."""
*/
class Conv2Impl : public BaseModule
{
    //def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
public:
    Conv2Impl(){};
    ~Conv2Impl(){};
    explicit Conv2Impl(int c1, int c2, int k = 3, int s = 1, std::optional<int> p = std::nullopt, int g = 1, int d = 1, std::string act = "True");
    
    void set_params(int inchannels, int number, std::vector<arg_complex>& args);
    // 简化下转参数的适应性，后续再修改
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override {return out_ch;} ;
    void check_args(std::vector<arg_complex>& args);

    void fuse_conv_and_bn(bool fused_ = true);
private:
    int in_ch;
    int out_ch;
    int k_size;
    int stride;
    int padding;

    bool fused = false;
    bool bfirst = true;
    
    void Init_Modules(int c1, int c2, int k = 3, int s = 1, std::optional<int> p = std::nullopt, int g = 1, int d = 1, std::string act = "True" );
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };

    torch::nn::AnyModule _act;

    torch::nn::Conv2d cv2 {nullptr};
    torch::nn::Conv2d fusedconv{ nullptr };
};
TORCH_MODULE(Conv2);


/*
    """Depth-wise convolution."""
*/
class DWConvImpl : public BaseModule
{
    //c1, c2, k=1, s=1, p=None, g=1, act=True
public:
    DWConvImpl() {};
    ~DWConvImpl() {};
    explicit DWConvImpl(int c1, int c2, int k = 1, int s = 1, int d = 1, std::string act = "True");

    void set_params(int inchannels, int number, std::vector<arg_complex>& args);
    // 简化下转参数的适应性，后续再修改
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override { return out_ch; };
    void check_args(std::vector<arg_complex>& args);

    void fuse_conv_and_bn(bool fused_ = true);
public:
    int in_ch;
    int out_ch;
    int k_size;
    int stride;
    int padding;

    bool fused = false;
    bool bfirst = true;

    void Init_Modules(int c1, int c2, int k = 1, int s = 1, int d = 1, std::string act = "True");
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };

    torch::nn::AnyModule _act;

    torch::nn::Conv2d fusedconv{ nullptr };
};
TORCH_MODULE(DWConv);

/*
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
  
    """Initialize Conv layer with given arguments including activation."""
*/
class LightConvImpl  : public BaseModule
{
public:    
    LightConvImpl() {};
    ~LightConvImpl() {};
    // def __init__(self, c1, c2, k=1, act=nn.ReLU()):
    explicit LightConvImpl(int c1, int c2, int k = 1, std::string act = "ReLU");

    void set_params(int inchannels, int number, std::vector<arg_complex>& args);
    // 简化下转参数的适应性，后续再修改
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override { return out_ch; };
    void check_args(std::vector<arg_complex>& args);  
public:  
    int in_ch;
    int out_ch;
    int k_size;
    int stride;
    int padding;

    Conv conv1;
    DWConv conv2;

    void Init_Modules(int c1, int c2, int k = 1, std::string act = "ReLU");
};
TORCH_MODULE(LightConv);

/*
    """Focus wh information into c-space."""
    """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
*/
class FocusImpl : public BaseModule
{
public:
    FocusImpl(){};
    ~FocusImpl(){};
    // def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
    explicit FocusImpl(int c1, int c2, int k = 1, int s = 1, 
        std::optional<int> p = std::nullopt, int g = 1, std::string act = "True");
    
    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    // 简化下转参数的适应性，后续再修改
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override {return out_ch;} ;
    void check_args(std::vector<arg_complex>& args);
private:
    int in_ch;
    int out_ch;
    int k_size;
    int stride;
    int padding;

    Conv conv;

    void Init_Modules(int c1, int c2, int k = 1, int s = 1,
        std::optional<int> p = std::nullopt, int g = 1, std::string act = "True");
};
TORCH_MODULE(Focus);

/*
    """Concatenate a list of tensors along dimension."""    """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
*/
class ConcatImpl : public BaseModule
{
public:
    ConcatImpl(){};
    ~ConcatImpl(){}; 

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override {return out_ch;} ;    
public:
    int in_ch;
    int out_ch;
    int dimen;
};
TORCH_MODULE(Concat);

/*
    """Concatenate a list of tensors along dimension."""    """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
*/
class GhostConvImpl : public BaseModule
{
public:
    GhostConvImpl(){};
    ~GhostConvImpl(){}; 
    // def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
    explicit GhostConvImpl(int c1, int c2, int k = 1, int s = 1, int g = 1, std::string act="True");
    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override {return out_ch;} ;    
public:
    int in_ch;
    int out_ch;
    int k_size;
    int stride;
    int padding;

    Conv cv1;
    Conv cv2;

    void Init_Modules(int c1, int c2, int k = 1, int s = 1, int g = 1, std::string act="True");    
};
TORCH_MODULE(GhostConv);

/*
    """Depth-wise transpose convolution module."""
*/
class DWConvTranspose2d : public torch::nn::ConvTranspose2d
{
    explicit DWConvTranspose2d(int64_t input_channels,
        int output_channels, int64_t k = 1, int s = 1, int p1 = 0, int p2 = 0)
        : torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(input_channels, output_channels, k)
            .stride(1).groups(std::gcd(input_channels, output_channels)).padding(p1).output_padding(p2))
    {
    };
};

/*
*/
class ConvTransposeImpl : public BaseModule
{
public:
    ConvTransposeImpl(){};
    ~ConvTransposeImpl(){};
    // def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
    explicit ConvTransposeImpl(int c1, int c2, int k = 2, int s = 2, int p0 = 0, bool bn = true, std::string act = "True");
    
    void set_params(int inchannels, int number, std::vector<arg_complex>& args);
    // 简化下转参数的适应性，后续再修改
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override {return out_ch;} ;
    void check_args(std::vector<arg_complex>& args);

    void fuse_conv_and_bn(bool fused_ = true);
public:
    int in_ch;
    int out_ch;
    int k_size;
    int stride;
    int padding;

    bool fused = false;
    bool bfirst = true;
    
    void Init_Modules(int c1, int c2, int k = 2, int s = 2, int p0 = 0, bool bn = true, std::string act = "True");
    torch::nn::ConvTranspose2d conv_transpose{ nullptr };

    torch::nn::AnyModule _bn;
    torch::nn::AnyModule _act;
};
TORCH_MODULE(ConvTranspose);

/*
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    "
*/
class ChannelAttentionImpl : public torch::nn::Module 
{
public:
    explicit ChannelAttentionImpl(int channels) {
        _pool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1));
        _fc = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 1).stride(1).padding(0).bias(true));
        _act = torch::nn::Sigmoid();

        register_module("pool", _pool);
        register_module("fc", _fc);
        register_module("act", _act);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return x * _act->forward(_fc->forward(_pool->forward(x)));
    }
public:
    torch::nn::AdaptiveAvgPool2d _pool{ nullptr };
    torch::nn::Conv2d _fc{ nullptr };
    torch::nn::Sigmoid _act{ nullptr };
};
TORCH_MODULE(ChannelAttention);

/*
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """
*/
class SpatialAttentionImpl : public torch::nn::Module
{
public:
    explicit SpatialAttentionImpl(int kernel_size = 7)
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

    torch::Tensor forward(torch::Tensor x)
    {
        auto [x_max_v, x_max_i] = torch::max(x, 1, true);
        auto x_mean = torch::mean(x, 1, true);
        return x * act->forward(cv1->forward(torch::cat({ x_mean, x_max_v }, 1)));
    }
public:
    torch::nn::Conv2d cv1{ nullptr };
    torch::nn::Sigmoid act{ nullptr };
};
TORCH_MODULE(SpatialAttention);

/*
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """
*/
class CBAMImpl : public torch::nn::Module {
public:
    CBAMImpl(int c1, int k = 7)
    {
        channel_attention = ChannelAttention(c1);
        spatial_attention = SpatialAttention(k);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return spatial_attention->forward(channel_attention->forward(x));
    }

public:
    ChannelAttention channel_attention{nullptr};
    SpatialAttention spatial_attention{nullptr};
};
TORCH_MODULE(CBAM);