#pragma once
/*
    对应着新版本中nn/block.py文件中的定义
    包含如下的modules
    "DFL",                  [√]
    "HGBlock",
    "HGStem",
    "SPP",                  [√]
    "SPPF",                 [√]
    "C1",
    "C2",
    "C3",                   [√]
    "C2f",                  [√]
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",           [√]
    "BottleneckCSP",        [√]
    "Proto",                [√]
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",                 [√]
    "C2fPSA",
    "C2PSA",                [√]
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",            [√]
    "PSA",
    "SCDown",
    "TorchVision",

    "A2C2f"                 [√]     Yolov12   ABlock AAttn

    "HyperACE",             [√] 
    "DownsampleConv",       [√]  
    "FullPAD_Tunnel",       [√] 
    "DSC3k2"                
*/


#include <BaseModel.h>
#include <torch/torch.h>

#include "conv.h"

/*
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """
*/
class DFLImpl : public torch::nn::Module 
{
public:
    explicit DFLImpl(int _c1 = 16);
    torch::Tensor forward(torch::Tensor x);
public:
    torch::nn::Conv2d conv{ nullptr };
    int c1;
};
TORCH_MODULE(DFL);

/*
    """Ultralytics YOLO models mask Proto module for segmentation models."""
*/
class ProtoImpl : public torch::nn::Module
{
public:
    explicit ProtoImpl(int c1, int c_ = 256, int c2 = 32);
    
    torch::Tensor forward(torch::Tensor x);
    int get_outchannels(){return out_ch;};
private:
    int in_ch;
    int out_ch;

    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    Conv cv3{ nullptr };
    torch::nn::ConvTranspose2d upsample{nullptr};
};
TORCH_MODULE(Proto);

/*
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""
*/
class SPPImpl : public BaseModule
{
public:
    SPPImpl() {};
    ~SPPImpl() {};

    void Init_Modules(int c1, int c2, std::vector<int> _k = {5, 9, 13});

    void  set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x)override;
    torch::Tensor forward(torch::Tensor x)override;
    int get_outchannels() override { return out_ch; };
public:
    int in_ch;
    int out_ch;
    std::vector<int> k;
    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    torch::nn::ModuleList m;
};
TORCH_MODULE(SPP);

/*
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
*/
class SPPFImpl : public BaseModule
{
public:
    SPPFImpl(){};
    ~SPPFImpl(){};
    // def __init__(self, c1: int, c2: int, k: int = 5):
    void Init_Modules(int c1, int c2, int k = 5);

    void  set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x)override;
    torch::Tensor forward(torch::Tensor x)override; 
    int get_outchannels() override {return out_ch;} ;    
public:
    int in_ch;
    int out_ch;
    int k;
    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    torch::nn::MaxPool2d m{nullptr};
};
TORCH_MODULE(SPPF);

/*
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
*/
class C2fImpl : public BaseModule
{
public:
    //    def __init__(self, c1: int, c2 : int, n : int = 1, shortcut : bool = False, g : int = 1, e : float = 0.5) :
    explicit C2fImpl(int c1, int c2, int n = 1, bool shortcut = false, int g = 1,  float e = 0.5);
    C2fImpl() {};
    ~C2fImpl() {};
    void check_args(std::vector<arg_complex>& args);

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x);
    torch::Tensor forward(torch::Tensor x);
    int get_outchannels() override { return out_ch; };
private:
    int in_ch;
    int out_ch;
    int number;
    int c;
    float expansion;    // python code, bottleneck定义没用传入值，直接用了1.0

    void Init_Modules(int c1, int c2, int n = 1, bool shortcut = false, int g = 1, float e = 0.5);
    torch::Tensor forward_split(torch::Tensor x);
    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    torch::nn::ModuleList m;
};
TORCH_MODULE(C2f);

/*
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
*/
class C3k2Impl : public BaseModule
{
public:
    //  self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True

    explicit C3k2Impl(int c1, int c2, int n = 1, bool c3k = false, float e = 0.5f, int g = 1, bool shortcut = true);
    C3k2Impl() {};
    ~C3k2Impl() {};
    void check_args(std::vector<arg_complex>& args);

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x);
    torch::Tensor forward(torch::Tensor x);
    int get_outchannels() override { return out_ch; };
private:
    int in_ch;
    int out_ch;
    int number;
    int c;
    float expansion;    // python code, bottleneck定义没用传入值，直接用了1.0
    bool use_c3k;
    void Init_Modules(int c1, int c2, int n = 1, bool c3k = false, float e = 0.5f, int g = 1, bool shortcut = true);
    torch::Tensor forward_split(torch::Tensor x);
    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    torch::nn::ModuleList m;
};
TORCH_MODULE(C3k2);

/*
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
*/
class C3kImpl : public torch::nn::Module
{
public:
    // def __init__(self, c1: int, c2 : int, n : int = 1, shortcut : bool = True, g : int = 1, e : float = 0.5, k : int = 3) :
    explicit C3kImpl(int c1, int c2, int n=1, bool shortcut=true, int g = 1, float e=0.5, int k_ = 3);
    torch::Tensor forward(torch::Tensor x){
        return cv3->forward(torch::cat({m->forward(cv1->forward(x)), cv2->forward(x)}, 1));
    }
public:    
    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    Conv cv3{ nullptr };
    torch::nn::Sequential m; 
};
TORCH_MODULE(C3k);

/*
    """CSP Bottleneck with 3 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
*/
class C3Impl : public BaseModule
{
public:
    explicit C3Impl(int c1, int c2, int n=1, bool shortcut=true, int g = 1, float e=0.5f);
    C3Impl(){};
    ~C3Impl(){};
    void check_args(std::vector<arg_complex>& args);

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x);
    torch::Tensor forward(torch::Tensor x);
    int get_outchannels() override {return out_ch;} ;
private:
    int in_ch;
    int out_ch;
    int number;
    float expansion;    

    void Init_Modules(int c1, int c2, int n=1, bool shortcut=true, int g = 1, float e=0.5f);

    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    Conv cv3{ nullptr };
    torch::nn::Sequential m; 
};
TORCH_MODULE(C3);

/*
    """C3 module with cross-convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
*/
class C3xImpl : public BaseModule
{
public:
    explicit C3xImpl(int c1, int c2, int n=1, bool shortcut=true, float e=0.5);
    C3xImpl(){};
    ~C3xImpl(){};
    void check_args(std::vector<arg_complex>& args);

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x);
    torch::Tensor forward(torch::Tensor x);
    int get_outchannels() override {return out_ch;} ;
private:
    int in_ch;
    int out_ch;
    int number;
    float expansion;    // python code, bottleneck定义没用传入值，直接用了1.0

    void Init_Modules(int c1, int c2, int n=1, bool shortcut=true, float e=0.5);

    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    Conv cv3{ nullptr };
    torch::nn::Sequential m; 
};
TORCH_MODULE(C3x);

/*
    """Standard bottleneck."""

    c1 (int): Input channels.
    c2 (int): Output channels.
    shortcut (bool): Whether to use shortcut connection.
    g (int): Groups for convolutions.
    k (tuple): Kernel sizes for convolutions.
    e (float): Expansion ratio.
*/
class BottleneckImpl : public torch::nn::Module
{
public:
    // self, c1 : int, c2 : int, shortcut : bool = True, 
    // g : int = 1, k : tuple[int, int] = (3, 3), e : float = 0.5

    explicit BottleneckImpl(int c1, int c2, bool shortcut = true, 
            int g = 1, std::tuple<int, int> k = std::make_tuple(3, 3),  
            float e = 0.5f);
    // 暂时还不需要，只由C3调用
    torch::Tensor forward(torch::Tensor x);
    int get_outchannels(){return out_ch;};
private:
    int in_ch;
    int out_ch;
    bool add_flag;
    float expansion;

    Conv cv1{nullptr};
    Conv cv2{nullptr};
};
TORCH_MODULE(Bottleneck);

/*
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""
        Initialize CSP Bottleneck.
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
*/
class BottleneckCSPImpl : public torch::nn::Module
{
public:
    //  def __init__(self, c1: int, c2: int, n: int = 1, 
    //      shortcut: bool = True, g: int = 1, e: float = 0.5):
    BottleneckCSPImpl(int c1, int c2, int n = 1, bool shortcut_ = true, int g=1, float e = 0.5f)
    {
        int c_ = int(float(c2 * e));
        cv1 = Conv(c1, c_, 1, 1);
        cv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c_, 1).stride(1).bias(false));
        cv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(c_, c_, 1).stride(1).bias(false));
        cv4 = Conv(2 * c_, c2, 1, 1);
        bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(2 * c_));
        act = torch::nn::SiLU();
        for (int i = 0; i < n; i++)
        {
            m->push_back(*(Bottleneck(c_, c_, shortcut_, g, std::make_tuple(3, 3), 1.0f)));
        }
        register_module("cv1", cv1);
        register_module("cv2", cv2);
        register_module("cv3", cv3);
        register_module("cv4", cv4);
        register_module("bn", bn);
        register_module("act", act);
        register_module("m", m);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        torch::Tensor y1 = cv3->forward(m->forward(cv1->forward(x)));
        torch::Tensor y2 = cv2->forward(x);
        return cv4->forward(act->forward(bn->forward(torch::cat({ y1, y2 }, 1))));
    }
public:
    Conv cv1{ nullptr };
    torch::nn::Conv2d cv2{ nullptr };
    torch::nn::Conv2d cv3{ nullptr };
    Conv cv4{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };
    torch::nn::SiLU act{ nullptr };
    torch::nn::Sequential m;
        
};
TORCH_MODULE(BottleneckCSP);

/*
    Area-attention module for YOLO models, providing efficient attention mechanisms.
*/
class AAttnImpl : public torch::nn::Module
{
public:
    explicit AAttnImpl(int _dim, int _num_heads, int _area = 1)
    {
        area = _area;
        num_heads = _num_heads;

        int h_dim = static_cast<int>(std::trunc(float(_dim) / float(_num_heads)));
        head_dim = h_dim;
        int all_head_dim = h_dim * num_heads;

        qkv_cv = Conv(_dim, all_head_dim * 3, 1, 1, std::nullopt, 1, 1, "False");
        proj_cv = Conv(all_head_dim, _dim, 1, 1, std::nullopt, 1, 1, "False");
        pe_cv = Conv(all_head_dim, _dim, 7, 1, 3, _dim, 1, "False");
        register_module("qkv", qkv_cv);
        register_module("proj", proj_cv);
        register_module("pe", pe_cv);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto B = x.size(0);
        auto C = x.size(1);
        auto H = x.size(2);
        auto W = x.size(3);
        auto N = H * W;

        auto qkv = qkv_cv->forward(x).flatten(2).transpose(1, 2);
        if (area > 1)
        {
            auto N_area = static_cast<int>(std::trunc(float(N) / float(area)));
            qkv = qkv.reshape({ B * area, N_area, C * 3 });
            B = qkv.size(0);
            N = qkv.size(1);
        }

        auto qkv_split = qkv.view({ B, N, num_heads, head_dim * 3 }).permute({ 0, 2, 3, 1 }).split(
            { head_dim, head_dim, head_dim }, 2);
        auto q = qkv_split[0];
        auto k = qkv_split[1];
        auto v = qkv_split[2];

        auto attn = (q.transpose(-2, -1).matmul(k)) * std::pow(float(head_dim), -0.5f);
        attn = attn.softmax(-1);
        x = v.matmul(attn.transpose(-2, -1));
        x = x.permute({ 0, 3, 1, 2 });
        v = v.permute({ 0, 3, 1, 2 });

        if (area > 1)
        {
            auto B_area = static_cast<int>(std::trunc(float(B) / float(area)));
            x = x.reshape({ B_area, int(N * area), C});
            v = v.reshape({ B_area, int(N * area), C });
            B = x.size(0);
            N = x.size(1);
        }

        x = x.reshape({ B, H, W, C }).permute({ 0, 3, 1, 2 }).contiguous();
        v = v.reshape({ B, H, W, C }).permute({ 0, 3, 1, 2 }).contiguous();
        x = x + pe_cv->forward(v);
        return proj_cv->forward(x);
    }

public:
    int area;
    int num_heads;
    int head_dim;

    Conv qkv_cv{ nullptr };
    Conv proj_cv{ nullptr };
    Conv pe_cv{ nullptr };
};
TORCH_MODULE(AAttn);

/*
    Area-attention block module for efficient feature extraction in YOLO models.
*/
class ABlockImpl : public torch::nn::Module
{
public:
    ABlockImpl(int dim, int num_heads, float mlp_ratio = 1.2, int area = 1)
    {
        //  std::cout << "ABlock input dim: " << dim << " num_heads: " << num_heads
        //      << " ratio: " << mlp_ratio << " area: " << area << "\n";

        attn = AAttn(dim, num_heads, area);
        mlp_hidden_dim = int(float(dim) * mlp_ratio);

        /*
        mlp->push_back(std::make_shared<ConvImpl>(dim, mlp_hidden_dim, 1));
        mlp->push_back(std::make_shared<ConvImpl>(mlp_hidden_dim, dim, 1, 1, -1, 1, 1, "False"));
        */

        mlp_0 = Conv(dim, mlp_hidden_dim, 1);
        mlp_1 = Conv(mlp_hidden_dim, dim, 1, 1, std::nullopt, 1, 1, "False"); 
        register_module("attn", attn);
        register_module("mlp-0", mlp_0);
        register_module("mlp-1", mlp_1);

        for (auto& sub_model : this->modules(false))
        {
            init_weights(*sub_model);
        }      
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = x + attn->forward(x);
        return x + mlp_1->forward(mlp_0->forward(x));
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
    void init_weights(torch::nn::Module& module) 
    {
        if (auto* conv = module.as<torch::nn::Conv2d>()) {
            torch::nn::init::normal_(conv->weight, 0.0, 0.02);
            conv->weight.data().clamp_(-2.0, 2.0);
            if (conv->bias.defined()) 
            {
                torch::nn::init::constant_(conv->bias, 0);
            }
            else
                std::cout << "conv bias not defined.\n";

        } //原代码只初始化了nn.Conv2d
        // else if (auto* bn = module.as<torch::nn::BatchNorm2d>()) {
        //     torch::nn::init::ones_(bn->weight);
        //     torch::nn::init::zeros_(bn->bias);
        // }
    }

public:
    int mlp_hidden_dim;

    AAttn attn{ nullptr };
    //torch::nn::Sequential mlp;
    Conv mlp_0{ nullptr };
    Conv mlp_1{ nullptr };
};
TORCH_MODULE(ABlock);

/*
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.
*/
class A2C2fImpl : public BaseModule
{
public:
    // [-1, 4, A2C2f, [512, True, 4/1/-1]] ==> c2 = 512,  n = 4,  a2 = True, area = 4/1/-1
    explicit A2C2fImpl(int c1, int c2, int n=1, bool a2 = true, 
        int area = 1, bool residual = false, float mlp_ratio = 2.0f,
        float e = 0.5, int g = 1, bool shortcut = true);
    A2C2fImpl(){};
    ~A2C2fImpl(){};
    void check_args(std::vector<arg_complex>& args);

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override {return out_ch;} ;
private:
    int in_ch;
    int out_ch;
    int number;
    bool a2_;
    bool residual_;
    
    void Init_Modules(int c1, int c2, int n=1, bool a2 = true, 
        int area = 1, bool residual = false, float mlp_ratio = 2.0f,
        float e = 0.5, int g = 1, bool shortcut = true);

    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    torch::nn::ModuleList m;
    torch::Tensor gamma{nullptr};
};
TORCH_MODULE(A2C2f);

class AttentionImpl : public torch::nn::Module
{
public:
    AttentionImpl(int dim, int _num_heads = 8, float _attn_ratio = 0.5f)
    {
        num_heads = _num_heads;
        head_dim = static_cast<int>(std::trunc(float(dim) / float(num_heads)));
        key_dim = int(float(head_dim) * _attn_ratio);
        scale = float(std::pow(key_dim, -0.5f));
        int nh_kd = key_dim * num_heads;
        int h = dim + nh_kd * 2;
        cv_qkv = Conv(dim, h, 1, 1, std::nullopt, 1, 1, "False");
        cv_proj = Conv(dim, dim, 1, 1, std::nullopt, 1, 1, "False");
        cv_pe = Conv(dim, dim, 3, 1, std::nullopt, dim, 1, "False");
        register_module("qkv", cv_qkv);
        register_module("proj", cv_proj);
        register_module("pe", cv_pe);
    }
    torch::Tensor forward(torch::Tensor x)
    {
        int B = x.size(0);
        int C = x.size(1);
        int H = x.size(2);
        int W = x.size(3);
        int N = H * W;

        auto qkv = cv_qkv->forward(x);
        auto qkv_split = qkv.view({ B, num_heads, key_dim * 2 + head_dim, N }).split(
            { key_dim, key_dim, head_dim }, 2);
        auto q = qkv_split[0];
        auto k = qkv_split[1];
        auto v = qkv_split[2];

        auto attn = (q.transpose(-2, -1).matmul(k)) * scale;
        attn = attn.softmax(-1);
        x = (v.matmul(attn.transpose(-2, -1)).view({ B, C, H, W })) +
            cv_pe->forward(v.reshape({ B, C, H, W }));
        x = cv_proj->forward(x);

        return x;
    }
private:
    int num_heads;
    int head_dim;
    int key_dim;
    float scale;

    Conv cv_qkv{ nullptr };
    Conv cv_proj{ nullptr };
    Conv cv_pe{ nullptr };
};
TORCH_MODULE(Attention);

/*
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """
*/
class PSABlockImpl : public torch::nn::Module
{
public:
    // def __init__(self, c: int, attn_ratio : float = 0.5, num_heads : int = 4, shortcut : bool = True)->None :

    explicit PSABlockImpl(int c, float attn_ratio = 0.5, int num_heads = 4, bool short_cut = true)
    {
        this->add_flag = short_cut;

        attn = Attention(c, num_heads, attn_ratio);

        // ffn = torch::nn::Sequential();
        ffn_cv0 = Conv(c, c*2, 1);
        ffn_cv1 = Conv(c * 2, c, 1, 1,std::nullopt, 1,1, "False");
        // ffn->push_back(ffn_cv0);
        // ffn->push_back(ffn_cv1);
        register_module("attn", attn);
        register_module("ffn-0", ffn_cv0);
        register_module("ffn-1", ffn_cv1);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        if (add_flag)
        {
            x = x + attn->forward(x);
            x = x + ffn_cv1->forward(ffn_cv0->forward(x));
        }
        else
        {
            x = attn->forward(x);
            x = ffn_cv1->forward(ffn_cv0->forward(x));
;
        }
        return x;
    }
private:
    Attention attn{ nullptr };
    // torch::nn::Sequential ffn{ nullptr };
    bool add_flag;
    Conv ffn_cv0{ nullptr };
    Conv ffn_cv1{ nullptr };
};
TORCH_MODULE(PSABlock);

/*
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

*/
class C2PSAImpl : public BaseModule
{
public:
    // def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
    explicit C2PSAImpl(int c1, int c2, int n = 1, float e = 0.5f);
    C2PSAImpl() {};
    ~C2PSAImpl() {};
    void check_args(std::vector<arg_complex>& args);

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override { return out_ch; };
private:
    int in_ch;
    int out_ch;
    int number;
    int c;

    void Init_Modules(int c1, int c2, int n = 1, float e = 0.5);

    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    torch::nn::Sequential m;

};
TORCH_MODULE(C2PSA);

// 2025-10-31 Add for v13

/*
    """
    An improved bottleneck block using depthwise separable convolutions (DSConv).

    This class implements a lightweight bottleneck module that replaces standard convolutions with depthwise
    separable convolutions to reduce parameters and computational cost.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to use a residual shortcut connection. The connection is only added if c1 == c2. Defaults to True.
        e (float, optional): Expansion ratio for the intermediate channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv layer. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv layer. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv layer. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSBottleneck module.

    Examples:
        >>> import torch
        >>> model = DSBottleneck(c1=64, c2=64, shortcut=True)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> output = model(x)
        >>> print(output.shape)
        t
 */
class DSBottleneckImpl : public torch::nn::Module
{
public:
    // def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
    DSBottleneckImpl(int c1, int c2, bool shortcut = true,
        float e = 0.5f, int k1 = 3, int k2 = 5, int d2 = 1)
    {
        int c_ = int(float(c2) * e);
        cv1 = DSConv(c1, c_, k1, 1, std::nullopt, 1);
        cv2 = DSConv(c_, c2, k2, 1, std::nullopt, d2);
        register_module("cv1", cv1);
        register_module("cv2", cv2);
        this->add = shortcut && c1 == c2;
    }

    torch::Tensor forward(torch::Tensor x) {
        auto y = cv2->forward(cv1->forward(x));
        if (add)
            return x + y;
        return y;
    }
public:
    DSConv cv1{ nullptr };
    DSConv cv2{ nullptr };
    bool add;
};
TORCH_MODULE(DSBottleneck);

/*
    """
    An improved C3k module using DSBottleneck blocks for lightweight feature extraction.

    This class extends the C3 module by replacing its standard bottleneck blocks with DSBottleneck blocks,
    which use depthwise separable convolutions.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of DSBottleneck blocks to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections within the DSBottlenecks. Defaults to True.
        g (int, optional): Number of groups for grouped convolution (passed to parent C3). Defaults to 1.
        e (float, optional): Expansion ratio for the C3 module's hidden channels. Defaults to 0.5.
        k1 (int, optional): Kernel size for the first DSConv in each DSBottleneck. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in each DSBottleneck. Defaults to 5.
        d2 (int, optional): Dilation for the second DSConv in each DSBottleneck. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k module (inherited from C3).

    Examples:
        >>> import torch
        >>> model = DSC3k(c1=128, c2=128, n=2, k1=3, k2=7)
        >>> x = torch.randn(2, 128, 64, 64)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 128, 64, 64])
    """
*/
class DSC3kImpl : public torch::nn::Module 
{
public:
    /*
        def __init__(
        self,
        c1,                
        c2,                 
        n=1,                
        shortcut=True,      
        g=1,                 
        e=0.5,              
        k1=3,               
        k2=5,               
        d2=1                 
    ):*/
    DSC3kImpl(int c1, int c2, int n = 1, bool shortcut = true, int g = 1,
        float e = 0.5, int k1 = 3, int k2 = 5, int d2 = 1) {

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

    torch::Tensor forward(torch::Tensor x) {
        return cv3->forward(torch::cat({
            m->forward(cv1->forward(x)),
            cv2->forward(x)
            }, 1));
    }

public:
    Conv cv1;
    Conv cv2;
    Conv cv3;
    torch::nn::Sequential m;

};
TORCH_MODULE(DSC3k);

/*
    """
    A module to fuse multi-scale features for the HyperACE block.

    This module takes a list of three feature maps from different scales, aligns them to a common
    spatial resolution by downsampling the first and upsampling the third, and then concatenates
    and fuses them with a convolution layer.

    Attributes:
        c_in (int): The number of channels of the input feature maps.
        channel_adjust (bool): Whether to adjust the channel count of the concatenated features.

    Methods:
        forward: Fuses a list of three multi-scale feature maps.

    Examples:
        >>> import torch
        >>> model = FuseModule(c_in=64, channel_adjust=False)
        >>> # Input is a list of features from different backbone stages
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
*/
class FuseModuleImpl : public torch::nn::Module
{
public:
    FuseModuleImpl(int c_in, bool channel_adjust)
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

    torch::Tensor forward(std::vector<torch::Tensor> x)
    {
        auto x1_ds = downsample->forward(x[0]);
        auto x3_up = upsample->forward(x[2]);
        auto x_cat = torch::cat({x1_ds, x[1], x3_up}, 1);
        auto out =  conv_out->forward(x_cat);
        return out;
    }

public:
    torch::nn::AvgPool2d downsample {nullptr};
    torch::nn::Upsample upsample {nullptr};    
    Conv conv_out;
};
TORCH_MODULE(FuseModule);

class AdaHyperedgeGenImpl : public torch::nn::Module
{
public:
    AdaHyperedgeGenImpl(int node_dim, int _num_hyperedges,
        int _num_heads = 4, float _dropout = 0.1,
        std::string _context = "both")
    {
        num_heads = _num_heads;
        num_hyperedges = _num_hyperedges;
        context = _context;
        head_dim = static_cast<int>(std::trunc(float(node_dim) / float(num_heads)));
        
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
            LOG(ERROR) << "context wrong： " << context;

        pre_head_proj = torch::nn::Linear(torch::nn::LinearOptions(
                node_dim, node_dim));

        // python __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        dropout_net = torch::nn::Dropout(torch::nn::DropoutOptions().p(_dropout));

        scaling = std::sqrt(this->head_dim);

        register_module("context_net", context_net);
        register_module("pre_head_proj", pre_head_proj);
        register_module("dropout", dropout_net);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        int B = x.size(0);
        int N = x.size(1);
        int D = x.size(2);
        torch::Tensor context_cat;

        if (context == "mean")
        {
            context_cat = x.mean(1);
        }    
        else if (context == "max")
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
            .view({ B, num_hyperedges, D });
        auto prototypes = prototype_base.unsqueeze(0) + prototype_offset;

        auto x_proj = pre_head_proj->forward(x);

        auto x_heads = x_proj.view({ B, N, num_heads, head_dim }).transpose(1, 2);

        auto proto_heads = prototypes.view({ B, num_hyperedges, num_heads,
                head_dim }).permute({0, 2, 1, 3});

        auto x_heads_flat = x_heads.reshape({ B * num_heads, N, head_dim });
        
        auto proto_heads_flat = proto_heads.reshape({ B * num_heads, num_hyperedges,
            head_dim }).transpose(1, 2);

        torch::Tensor logits = torch::bmm(x_heads_flat, proto_heads_flat) / this->scaling;
        logits = logits.view({ B, num_heads, N, num_hyperedges }).mean(1);

        logits = dropout_net->forward(logits);

        return torch::nn::functional::softmax(logits, torch::nn::functional::SoftmaxFuncOptions(1));
    }

public:
    int num_hyperedges;
    int num_heads;
    float dropout;
    std::string context;

    int head_dim;

    torch::Tensor prototype_base;
    torch::nn::Linear context_net{ nullptr };
    torch::nn::Linear pre_head_proj{ nullptr };
    torch::nn::Dropout dropout_net{ nullptr };

    double scaling;
};
TORCH_MODULE(AdaHyperedgeGen);
/*
*  libtorch中没有GELU，可以直接计算，为保持代码与python的一致，写成类
*/
class GELUImpl : public torch::nn::Module
{
public:
    GELUImpl() {};
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

    torch::Tensor forward(torch::Tensor x)
    {
        //精度要求不高的近似解如下：
        //return 0.5 * x * (1 + torch::tanh(std::sqrt(2 / M_PI) * (x + 0.044715 * torch::pow(x, 3)));
        // 标准公式代码如下：
        return x * 0.5 * (1 + torch::erf(x * std::sqrt(2)));
        
    }
};
TORCH_MODULE(GELU);

class AdaHGConvImpl : public torch::nn::Module 
{
public:
    // def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, 
    // dropout=0.1, context="both"):
    AdaHGConvImpl(int embed_dim, int num_hyperedge = 16, int num_heads = 4,
        float dropout = 0.1, std::string context = "both")
    {
        edge_generator = AdaHyperedgeGen(embed_dim, num_hyperedge, num_heads,
            dropout, context);
        edge_proj = torch::nn::Sequential(
            torch::nn::Linear(embed_dim, embed_dim),
            GELU()
        );

        node_proj = torch::nn::Sequential(
            torch::nn::Linear(embed_dim, embed_dim),
            GELU()
        );

        register_module("edge_generator", edge_generator);
        register_module("edge_proj", edge_proj);
        register_module("note_proj", node_proj);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto A = edge_generator->forward(x);
        
        auto He = torch::bmm(A.transpose(1, 2), x);
        
        He = edge_proj->forward(He);
        auto X_new = torch::bmm(A, He);

        X_new = node_proj->forward(X_new);
        return X_new + x;
    }

public:
    AdaHyperedgeGen edge_generator{ nullptr };
    torch::nn::Sequential node_proj;
    torch::nn::Sequential edge_proj;
};
TORCH_MODULE(AdaHGConv);

/*
    """
    A wrapper module for applying adaptive hypergraph convolution to 4D feature maps.

    This class makes the hypergraph convolution compatible with standard CNN architectures. It flattens a
    4D input tensor (B, C, H, W) into a sequence of vertices (tokens), applies the AdaHGConv layer to
    model high-order correlations, and then reshapes the output back into a 4D tensor.

    Attributes:
        embed_dim (int): The feature dimension of the vertices (equivalent to input channels C).
        num_hyperedges (int, optional): The number of hyperedges for the underlying AdaHGConv. Defaults to 16.
        num_heads (int, optional): The number of attention heads for the underlying AdaHGConv. Defaults to 8.
        dropout (float, optional): The dropout rate for the underlying AdaHGConv. Defaults to 0.1.
        context (str, optional): The context type for the underlying AdaHGConv. Defaults to "both".

    Methods:
        forward: Processes a 4D feature map through the adaptive hypergraph computation layer.

    Examples:
        >>> import torch
        >>> model = AdaHGComputation(embed_dim=64, num_hyperedges=8, num_heads=4)
        >>> x = torch.randn(2, 64, 32, 32) # (B, C, H, W)
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
*/
class AdaHGComputationImpl : public torch::nn::Module
{
public:
 //  def __init__(self, embed_dim, num_hyperedges=16, num_heads=8, dropout=0.1, context="both"):
    AdaHGComputationImpl(int embed_dim, int num_hyperedges = 16, int num_heads = 8,
        float dropout = 0.1, std::string context = "both") 
    {
        hgnn = AdaHGConv(embed_dim, num_hyperedges, num_heads,
            dropout, context);
        register_module("hgnn", hgnn);
    }

    torch::Tensor forward(torch::Tensor x) {
        int B = x.size(0);
        int C = x.size(1);
        int H = x.size(2);
        int W = x.size(3);
        auto tokens = x.flatten(2).transpose(1, 2);
        tokens = hgnn->forward(tokens);

        auto x_out = tokens.transpose(1, 2).view({B, C, H, W});
        return x_out;
    }
public:
    AdaHGConv hgnn{ nullptr };
};
TORCH_MODULE(AdaHGComputation);
/*
"""
A CSP - style block integrating Adaptive Hypergraph Computation(C3AH).

The input feature map is split into two paths.
One path is processed by the AdaHGComputation module to model high - order correlations, while the other
serves as a shortcut.The outputs are then concatenated to fuse features.

Attributes:
c1(int) : Number of input channels.
c2(int) : Number of output channels.
e(float, optional) : Expansion ratio for the hidden channels.Defaults to 1.0.
num_hyperedges(int, optional) : The number of hyperedges for the internal AdaHGComputation.Defaults to 8.
context(str, optional) : The context type for the internal AdaHGComputation.Defaults to "both".

Methods :
    forward : Performs a forward pass through the C3AH module.

    Examples :
    >> > import torch
    >> > model = C3AH(c1 = 64, c2 = 128, num_hyperedges = 8)
    >> > x = torch.randn(2, 64, 32, 32)
    >> > output = model(x)
    >> > print(output.shape)
    torch.Size([2, 128, 32, 32])
    """
*/
class C3AHImpl : public torch::nn::Module {
public:
    C3AHImpl(int c1, int c2, float e = 1.0, int num_hyperedges = 8, std::string context = "bost")
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

    torch::Tensor forward(torch::Tensor x)
    {
        return cv3->forward(torch::cat({
            m_adahg->forward(cv1->forward(x)),
                cv2->forward(x)}, 1));
    }

public:
    Conv cv1;
    Conv cv2;
    Conv cv3;
    AdaHGComputation m_adahg{nullptr};
};
TORCH_MODULE(C3AH);
/*
    """
    Hypergraph-based Adaptive Correlation Enhancement (HyperACE).

    This is the core module of YOLOv13, designed to model both global high-order correlations and
    local low-order correlations. It first fuses multi-scale features, then processes them through parallel
    branches: two C3AH branches for high-order modeling and a lightweight DSConv-based branch for
    low-order feature extraction.

    Attributes:
        c1 (int): Number of input channels for the fuse module.
        c2 (int): Number of output channels for the entire block.
        n (int, optional): Number of blocks in the low-order branch. Defaults to 1.
        num_hyperedges (int, optional): Number of hyperedges for the C3AH branches. Defaults to 8.
        dsc3k (bool, optional): If True, use DSC3k in the low-order branch; otherwise, use DSBottleneck. Defaults to True.
        shortcut (bool, optional): Whether to use shortcuts in the low-order branch. Defaults to False.
        e1 (float, optional): Expansion ratio for the main hidden channels. Defaults to 0.5.
        e2 (float, optional): Expansion ratio within the C3AH branches. Defaults to 1.
        context (str, optional): Context type for C3AH branches. Defaults to "both".
        channel_adjust (bool, optional): Passed to FuseModule for channel configuration. Defaults to True.

    Methods:
        forward: Performs a forward pass through the HyperACE module.

    Examples:
        >>> import torch
        >>> model = HyperACE(c1=64, c2=256, n=1, num_hyperedges=8)
        >>> x_list = [torch.randn(2, 64, 64, 64), torch.randn(2, 64, 32, 32), torch.randn(2, 64, 16, 16)]
        >>> output = model(x_list)
        >>> print(output.shape)
        torch.Size([2, 256, 32, 32])
    """
    */
class HyperACEImpl : public BaseModule
{
public:
    //def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=True):
    explicit HyperACEImpl(int c1, int c2, int n = 1, int num_hyperedges = 8, 
        bool dsc3k = true,
        bool shortcut = false,
        float e1 = 0.5f, float e2 = 1.0f, std::string context = "both", bool channel_adjust = true)
        {
            Init_Modules(c1, c2, n, num_hyperedges, dsc3k, shortcut, e1, e2, context, channel_adjust);
        };
    HyperACEImpl() {};
    ~HyperACEImpl() {};

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override { return out_ch; };
private:
    int in_ch;
    int out_ch;
    int number;
    int c;
    bool b_use_dsc3k;

    void Init_Modules(int c1, int c2, int n = 1, int num_hyperedges = 8, 
        bool dsc3k = true,
        bool shortcut = false,
        float e1 = 0.5f, float e2 = 1.0f, std::string context = "both", bool channel_adjust = true);

    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    torch::nn::ModuleList m;

    FuseModule fuse{ nullptr };
    C3AH branch1{ nullptr };
    C3AH branch2{ nullptr };
    
};
TORCH_MODULE(HyperACE);

class DownsampleConvImpl : public BaseModule
{
public:
    // def __init__(self, in_channels, channel_adjust=True):
    explicit DownsampleConvImpl(int in_channels, bool _channel_adjust = true)
    {
        Init_Modules(in_channels, _channel_adjust);
    }
    DownsampleConvImpl() {};
    ~DownsampleConvImpl() {};

    void set_params(int in_channels, int number, std::vector<arg_complex>& args)
    {
        int args_size = args.size();
        if (args_size == 1)
            Init_Modules(in_channels, std::get<bool>(args[0]));
        else
            LOG(WARNING) << "default args is empty() == true";
    }
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;
    int get_outchannels() override { return out_ch; };
private:
    int in_ch;
    int out_ch;
    int number;

    bool b_channel_adjust;
    torch::nn::AvgPool2d downsample{ nullptr };
    Conv channel_adjust;

    void Init_Modules(int in_channels, bool _channel_adjust = true); 

};
TORCH_MODULE(DownsampleConv);

class FullPAD_TunnelImpl : public BaseModule
{
public:
    // def __init__(self, in_channels):
    explicit FullPAD_TunnelImpl(int in_channels)
    {
        Init_Modules(in_channels);
    }
    FullPAD_TunnelImpl() {};
    ~FullPAD_TunnelImpl() {};

    void set_params(int in_channels, int number, std::vector<arg_complex>& args)
    {
        int args_size = args.size();
        if (args_size == 1)
            Init_Modules(in_channels);
        else
            LOG(WARNING) << "default args is empty() == true";
    }
    torch::Tensor forward(std::vector<torch::Tensor> x) {
        return x[0] + gate * x[1];
    }
    torch::Tensor forward(torch::Tensor x) {
        LOG(ERROR) << "need input vecotr.";
        return x;
    }
    int get_outchannels() override { return out_ch; };
private:
    int in_ch;
    int out_ch;
    int number;

    torch::Tensor gate;

    void Init_Modules(int in_channels) {
        in_ch = in_channels;
        out_ch = in_channels;

        this->gate = register_parameter("gate", torch::tensor(0.0));
    }

};
TORCH_MODULE(FullPAD_Tunnel);

/*
    """
    An improved C3k2 module that uses lightweight depthwise separable convolution blocks.

    This class redesigns C3k2 module, replacing its internal processing blocks with either DSBottleneck
    or DSC3k modules.

    Attributes:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of internal processing blocks to stack. Defaults to 1.
        dsc3k (bool, optional): If True, use DSC3k as the internal block. If False, use DSBottleneck. Defaults to False.
        e (float, optional): Expansion ratio for the C2f module's hidden channels. Defaults to 0.5.
        g (int, optional): Number of groups for grouped convolution (passed to parent C2f). Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connections in the internal blocks. Defaults to True.
        k1 (int, optional): Kernel size for the first DSConv in internal blocks. Defaults to 3.
        k2 (int, optional): Kernel size for the second DSConv in internal blocks. Defaults to 7.
        d2 (int, optional): Dilation for the second DSConv in internal blocks. Defaults to 1.

    Methods:
        forward: Performs a forward pass through the DSC3k2 module (inherited from C2f).

    Examples:
        >>> import torch
        >>> # Using DSBottleneck as internal block
        >>> model1 = DSC3k2(c1=64, c2=64, n=2, dsc3k=False)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output1 = model1(x)
        >>> print(f"With DSBottleneck: {output1.shape}")
        With DSBottleneck: torch.Size([2, 64, 128, 128])
        >>> # Using DSC3k as internal block
        >>> model2 = DSC3k2(c1=64, c2=64, n=1, dsc3k=True)
        >>> output2 = model2(x)
        >>> print(f"With DSC3k: {output2.shape}")
        With DSC3k: torch.Size([2, 64, 128, 128])
    """
*/
class DSC3k2Impl : public BaseModule
{
public:
//      def __init__(self, c1, c2, n = 1, dsc3k = False, 
//          e = 0.5, g = 1, shortcut = True, k1 = 3, k2 = 7,  d2 = 1 ):
    explicit DSC3k2Impl(int c1, int c2, int n = 1, bool _dsc3k = false, float e = 0.5,
        int g = 1, bool shortcut = true, int k1 = 3, int k2 = 7, int d2 = 1)
    {
        Init_Modules(c1, c2, n, _dsc3k, e, g, shortcut, k1, k2, d2);
    }
    DSC3k2Impl() {};
    ~DSC3k2Impl() {};

    void set_params(int in_channels, int number, std::vector<arg_complex>& args)
    {
        int args_size = args.size();
        if (args_size == 3)
            Init_Modules(in_channels, std::get<int>(args[0]),
                number, 
                std::get<bool>(args[1]),
                std::get<float>(args[2])
            );
        else if(args_size == 2)
            Init_Modules(in_channels, std::get<int>(args[0]),
                number, 
                std::get<bool>(args[1]));
        else
            LOG(WARNING) << "default args support 3";
    }
    torch::Tensor forward(std::vector<torch::Tensor> x) {
        LOG(ERROR) << "DSC3k2Impl forward not support vector input";
        return x[0];
    }
    torch::Tensor forward(torch::Tensor x);
    int get_outchannels() override { return out_ch; };
private:
    int in_ch;
    int out_ch;
    int number;

    bool b_dsc3k;

    Conv cv1;
    Conv cv2;
    torch::nn::ModuleList m;

    void Init_Modules(int c1, int c2, int n = 1, bool _dsc3k = false, float e = 0.5,
        int g = 1, bool shortcut = true, int k1 = 3, int k2 = 7, int d2 = 1);

};
TORCH_MODULE(DSC3k2);