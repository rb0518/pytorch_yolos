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

    A2C2f                   [√]     Yolov12   ABlock AAttn
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

        qkv_cv = Conv(_dim, all_head_dim * 3, 1, 1, -1, 1, 1, "False");
        proj_cv = Conv(all_head_dim, _dim, 1, 1, -1, 1, 1, "False");
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
        auto ret_x = v.matmul(attn.transpose(-2, -1));
        ret_x = ret_x.permute({ 0, 3, 1, 2 });
        v = v.permute({ 0, 3, 1, 2 });

        if (area > 1)
        {
            auto B_area = static_cast<int>(std::trunc(float(B) / float(area)));
            ret_x = ret_x.reshape({ B_area, int(N * area), C});
            v = v.reshape({ B_area, int(N * area), C });
            B = ret_x.size(0);
            N = ret_x.size(1);
        }

        ret_x = ret_x.reshape({ B, H, W, C }).permute({ 0, 3, 1, 2 }).contiguous();
        v = v.reshape({ B, H, W, C }).permute({ 0, 3, 1, 2 }).contiguous();
        ret_x = ret_x + pe_cv->forward(v);
        return proj_cv->forward(ret_x);
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
        // std::cout << "ABlock input dim: " << dim << " num_heads: " << num_heads
        //     << " ratio: " << mlp_ratio << " area: " << area << "\n";

        attn = AAttn(dim, num_heads, area);
        mlp_hidden_dim = int(float(dim) * mlp_ratio);

        /*
        mlp->push_back(std::make_shared<ConvImpl>(dim, mlp_hidden_dim, 1));
        mlp->push_back(std::make_shared<ConvImpl>(mlp_hidden_dim, dim, 1, 1, -1, 1, 1, "False"));
        */

        mlp_0 = Conv(dim, mlp_hidden_dim, 1);
        mlp_1 = Conv(mlp_hidden_dim, dim, 1, 1, -1, 1, 1, "False"); 
        register_module("attn", attn);
        //register_module("mpl", mlp);
        register_module("mlp-0", mlp_0);
        register_module("mlp-1", mlp_1);

        for (auto& sub_model : this->modules(false))
        {
            init_weights(*sub_model);
        }      
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto y = x + attn->forward(x);
        return y + mlp_1->forward(mlp_0->forward(y));
    }

    void truncated_normal_(torch::Tensor& tensor, float mean=0, float std=0.02) 
    {
        auto normal_tensor = torch::randn_like(tensor).mul_(std).add_(mean);
        auto mask = (normal_tensor < 2) & (normal_tensor > -2);
        tensor.copy_(normal_tensor.masked_fill_(~mask, 0));
    }

    void init_weights(torch::nn::Module& module) 
    {
        if (auto* conv = module.as<torch::nn::Conv2d>()) {
            torch::nn::init::normal_(conv->weight, 0.0f, 0.2f);
            if (conv->bias.defined()) 
                torch::nn::init::zeros_(conv->bias);
        }
        else if (auto* bn = module.as<torch::nn::BatchNorm2d>()) {
            torch::nn::init::ones_(bn->weight);
            torch::nn::init::zeros_(bn->bias);
        }
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