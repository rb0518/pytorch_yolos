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
    "Index"
    "DSConv"                [√]
*/


#include <BaseModel.h>
#include <torch/torch.h>

int autopad(int k, std::optional<int> p = std::nullopt, int d = 1);

/*
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
*/
class ConvImpl : public BaseModule
{
    // def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
public:
    ConvImpl(){};
    ConvImpl(int in_channels, int number, std::vector<arg_complex>& args) ;

    explicit ConvImpl(int c1, int c2, int k = 1, int s = 1, std::optional<int> p = std::nullopt, int g = 1, int d = 1, std::string act = "True");
    
    virtual void set_params(int in_channels, int number, std::vector<arg_complex>& args) ;
    virtual int get_outchannels() ;

    torch::Tensor forward(torch::Tensor x) ;

    void fuse_conv_and_bn(bool fused_ = true);
protected:
    int _in_channel;
    int _out_channel;
    int _kernel_size;
    int _stride;
    std::optional<int> _padding;
    int _groups;
    int _dilation;
    std::string _act;

    bool fused = false;

    void Init_Modules(int c1, int c2, int k = 1, int s = 1, std::optional<int> p = std::nullopt, int g = 1, int d = 1, std::string act = "True");
public:    
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };
    torch::nn::AnyModule active_fn;

    torch::nn::Conv2d fusedconv{ nullptr };
};
TORCH_MODULE(Conv);

/*
    """Simplified RepConv module with Conv fusing."""
*/
class Conv2Impl : public ConvImpl
{
public:
    Conv2Impl(){};

    //def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
    explicit Conv2Impl(int c1, int c2, int k = 3, int s = 1, std::optional<int> p = std::nullopt, int g = 1, int d = 1, std::string act = "True");
    
    virtual void set_params(int inchannels, int number, std::vector<arg_complex>& args) ;
    torch::Tensor forward(torch::Tensor x) ;

protected:
   
    void Init_Modules(int c1, int c2, int k = 3, int s = 1, std::optional<int> p = std::nullopt, int g = 1, int d = 1, std::string act = "True" );
public:  

    torch::nn::Conv2d cv2 {nullptr};
};
TORCH_MODULE(Conv2);


/*
    """Depth-wise convolution."""
    在rtdetr-[lx].yaml中定义过，5个参数
*/
class DWConvImpl : public ConvImpl
{
public:
    DWConvImpl() {};
    
    //     def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
    explicit DWConvImpl(int c1, int c2, int k = 1, int s = 1, int d = 1, std::string act = "True");
   
    virtual void set_params(int inchannels, int number, std::vector<arg_complex>& args) ;

    torch::Tensor forward(torch::Tensor x) ;
protected:
    void Init_Modules(int c1, int c2, int k = 1, int s = 1, int d = 1, std::string act = "True");
public:
};
TORCH_MODULE(DWConv);

/*
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
  
    """Initialize Conv layer with given arguments including activation."""
    未在任何配置文件中出现，属直接调用
*/
class LightConvImpl  : public torch::nn::Module
{
public:    
    // def __init__(self, c1, c2, k=1, act=nn.ReLU()):
    explicit LightConvImpl(int c1, int c2, int k = 1, std::string act = "ReLU");
    torch::Tensor forward(torch::Tensor x) ;

public:  
    Conv conv1;
    DWConv conv2;
};
TORCH_MODULE(LightConv);

/*
    """Focus wh information into c-space."""
    """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
    后续已经未在配置中出现，暂时保留
*/
class FocusImpl : public BaseModule
{
public:
    FocusImpl(){};
    FocusImpl(int in_channels, int number, std::vector<arg_complex>& args);

    // def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
    explicit FocusImpl(int c1, int c2, int k = 1, int s = 1, 
        std::optional<int> p = std::nullopt, int g = 1, std::string act = "True");
    
    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    int get_outchannels();

    torch::Tensor forward(torch::Tensor x);
private:
    int _out_channel;

    Conv conv{ nullptr };

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
    ConcatImpl(int in_channels, int number, std::vector<arg_complex>& args);

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    int get_outchannels() ;

    torch::Tensor forward(std::vector<torch::Tensor> x) ;
public:
    int _dimen;
};
TORCH_MODULE(Concat);

/*
    """Concatenate a list of tensors along dimension."""    """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
    在v8中配置中实用，暂时未用到
*/
class GhostConvImpl : public BaseModule
{
public:
    GhostConvImpl(){};
    GhostConvImpl(int in_channels, int number, std::vector<arg_complex>& args);

    // def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
    explicit GhostConvImpl(int c1, int c2, int k = 1, int s = 1, int g = 1, std::string act="True");
    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    int get_outchannels() ;    

    torch::Tensor forward(torch::Tensor x) ;
protected:
    int _out_channel;

    void Init_Modules(int c1, int c2, int k = 1, int s = 1, int g = 1, std::string act = "True");
public:
    Conv cv1;
    Conv cv2;
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
* """Convolution transpose 2d layer."""
* 在v6版本中实用，暂时未实现，先保留， 4 个参数
*/
class ConvTransposeImpl : public BaseModule
{
public:
    ConvTransposeImpl(){};
    ConvTransposeImpl(int in_channels, int number, std::vector<arg_complex>& args);

    // def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
    explicit ConvTransposeImpl(int c1, int c2, int k = 2, int s = 2, int p = 0, bool bn = true, std::string act = "True");
    
    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    int get_outchannels() ;

    torch::Tensor forward(torch::Tensor x) ;

    void fuse_conv_and_bn(bool fused_ = true);

protected:
    int _out_channel;

    void Init_Modules(int c1, int c2, int k = 2, int s = 2, int p = 0, bool bn = true, std::string act = "True");

public:
    torch::nn::ConvTranspose2d conv_transpose{ nullptr };

    torch::nn::AnyModule anymodule_bn;   // 用AnyModule，不用forward和forward_fuse
    torch::nn::AnyModule anymodule_act;
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
    explicit ChannelAttentionImpl(int channels);
    torch::Tensor forward(torch::Tensor x);
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
    explicit SpatialAttentionImpl(int kernel_size = 7);
    torch::Tensor forward(torch::Tensor x);
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
    CBAMImpl(int c1, int kernel_size = 7);
    torch::Tensor forward(torch::Tensor x);

public:
    ChannelAttention channel_attention{nullptr};
    SpatialAttention spatial_attention{nullptr};
};
TORCH_MODULE(CBAM);

// 2025-10-31 Add for v13
/*
    """The Basic Depthwise Separable Convolution."""
*/
class DSConvImpl : public BaseModule
{
   // def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
public:
    DSConvImpl(){};
    DSConvImpl(int in_channels, int number, std::vector<arg_complex>& args);

    explicit DSConvImpl(int c_in, int c_out, int k = 3, int s = 1,
        std::optional<int> p = std::nullopt, int d = 1, bool bias = false);
    
    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    int get_outchannels() ;   
    
    torch::Tensor forward(torch::Tensor x);

protected:
    int _out_channel;

    void Init_Modules(int c_in, int c_out, int k = 3, int s = 1, std::optional<int> p = std::nullopt, int d = 1, bool bias = false);

public:
    torch::nn::Conv2d dw{ nullptr };
    torch::nn::Conv2d pw{ nullptr };

    torch::nn::BatchNorm2d bn{ nullptr };
    torch::nn::SiLU act{nullptr};
};
TORCH_MODULE(DSConv);