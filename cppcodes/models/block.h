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
#include "common.h"
/*
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """
*/
class DFLImpl : public torch::nn::Module 
{
public:
    explicit DFLImpl(int c1 = 16);
    torch::Tensor forward(torch::Tensor x);
public:
    torch::nn::Conv2d conv{ nullptr };
    int _c1;
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

public:
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
    SPPImpl(int in_channels, int number, std::vector<arg_complex>& args);
    // int get_outchannels()  ;

    torch::Tensor forward(torch::Tensor x) ;

protected:
    void Init_Modules(int c1, int c2, std::vector<int> k = { 5, 9, 13 });

public:
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
    using Conf_Params = std::tuple<int, int>;
    Conf_Params get_defaults() {
        return std::make_tuple(0, 5);
    }

    SPPFImpl(int in_channels, int number, std::vector<arg_complex>& args);

    // def __init__(self, c1: int, c2: int, k: int = 5):
    void Init_Modules(int c1, int c2, int k = 5);
    // int get_outchannels()  ;  

    torch::Tensor forward(torch::Tensor x) ; 

public:
    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    torch::nn::MaxPool2d m{nullptr};
};
TORCH_MODULE(SPPF);

/*
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
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
    v8开始调用
*/
class C2fImpl : public BaseModule
{
public:
    using Conf_Params = std::tuple<int, bool, int, float>;
    Conf_Params get_defaults() {
        return std::make_tuple(0, false, 1, 0.5f);
    }

    C2fImpl(int in_channels, int number, std::vector<arg_complex>& args);
    //    def __init__(self, c1: int, c2 : int, n : int = 1, shortcut : bool = False, g : int = 1, e : float = 0.5) :
    explicit C2fImpl(int c1, int c2, int n = 1, bool shortcut = false, int g = 1,  float e = 0.5);

    // int get_outchannels()  ;

    torch::Tensor forward(torch::Tensor x);
    torch::Tensor forward_split(torch::Tensor x);

protected:
    int _c;

public:
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
    using Conf_Params = std::tuple<int, bool, float, int, bool>;
    Conf_Params get_defaults() {
        return std::make_tuple(0, false, 0.5f, 1, true);
    }

    C3k2Impl(int in_channels, int number, std::vector<arg_complex>& args);
    //  self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    explicit C3k2Impl(int c1, int c2, int n = 1, bool c3k = false, float e = 0.5f, int g = 1, bool shortcut = true);


    torch::Tensor forward(torch::Tensor x);
    torch::Tensor forward_split(torch::Tensor x);
public:
    bool _b_c3k;
    int _c;
protected:
    void Init(int c1, int c2, int n = 1, bool c3k = false, float e = 0.5f, int g = 1, bool shortcut = true);

public:
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
    v5 P6版本中代码，保留
*/
class C3Impl : public BaseModule
{
public:
    using Conf_Params = std::tuple<int, bool, int, float>;
    Conf_Params get_defaults(){
        return std::make_tuple(0, true, 1, 0.5f);
    }

    C3Impl(int in_channels, int number, std::vector<arg_complex>& args);

    // def __init__(self, c1, c2, n = 1, shortcut = True, g = 1, e = 0.5) : # ch_in, ch_out, number, shortcut, groups, expansion
    explicit C3Impl(int c1, int c2, int n=1, bool shortcut=true, int g = 1, float e=0.5f);

    // int get_outchannels()  ;

    torch::Tensor forward(torch::Tensor x);

public:
    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    Conv cv3{ nullptr };
    torch::nn::Sequential m; 
};
TORCH_MODULE(C3);

/*
    """C3 module with cross-convolutions."""
    未见配置文件中有调用
*/
class C3xImpl : public torch::nn::Module
{
public:
    // def __init__(self, c1, c2, n = 1, shortcut = True, g = 1, e = 0.5) : # ch_in, ch_out, number, shortcut, groups, expansion
    explicit C3xImpl(int c1, int c2, int n=1, bool shortcut=true, int g= 1, float e=0.5);

    torch::Tensor forward(torch::Tensor x);

public:
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
    只在v3版本中调用，暂时不做调用接口处理
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

private:
    bool _b_add;
public:
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
    BottleneckCSPImpl(int c1, int c2, int n = 1, bool shortcut_ = true, int g = 1, float e = 0.5f);
    torch::Tensor forward(torch::Tensor x);

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
    // def __init__(self, dim: int, num_heads: int, area: int = 1):
    explicit AAttnImpl(int dim, int num_heads, int area = 1);
    torch::Tensor forward(torch::Tensor x);

protected:
    int _area;
    int _num_heads;
    int _head_dim;
public:
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
    // def __init__(self, dim: int, num_heads : int, mlp_ratio : float = 1.2, area : int = 1) :
    ABlockImpl(int dim, int num_heads, float mlp_ratio = 1.2, int area = 1);
    torch::Tensor forward(torch::Tensor x);

    void init_weights(torch::nn::Module& module);

public:
    AAttn attn{ nullptr };
    torch::nn::Sequential mlp;
};
TORCH_MODULE(ABlock);

/*
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """
*/
class A2C2fImpl : public BaseModule
{
public:
    using Conf_Params = std::tuple<int, bool, int, bool, float, float, int, bool>;
    Conf_Params get_defaults() {
        return std::make_tuple(0, true, 1, false, 2.0f, 0.5f, 1, true);
    }

    A2C2fImpl(int in_channels, int number, std::vector<arg_complex>& args);

    // [-1, 4, A2C2f, [512, True, 4/1/-1]] ==> c2 = 512,  n = 4,  a2 = True, area = 4/1/-1
    explicit A2C2fImpl(int c1, int c2, int n=1, bool a2 = true, 
        int area = 1, bool residual = false, float mlp_ratio = 2.0,
        float e = 0.5, int g = 1, bool shortcut = true);

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    // int get_outchannels()  ;

    torch::Tensor forward(torch::Tensor x)  ;
private:
    bool _a2;
    
    void Init_Modules(int c1, int c2, int n=1, bool a2 = true, 
        int area = 1, bool residual = false, float mlp_ratio = 2.0,
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
    //  def __init__(self, dim, num_heads=8, attn_ratio=0.5):
    AttentionImpl(int dim, int num_heads = 8, float attn_ratio = 0.5f);
    torch::Tensor forward(torch::Tensor x);
private:
    int _num_heads;
    int _head_dim;
    int _key_dim;
    float _scale;
public:
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
    explicit PSABlockImpl(int c, float attn_ratio = 0.5, int num_heads = 4, bool short_cut = true);

    torch::Tensor forward(torch::Tensor x);
private:
    bool _b_add;

public:
    torch::nn::Sequential ffn{ nullptr };
    Attention attn{ nullptr };
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
    C2PSAImpl() {};
    C2PSAImpl(int in_channels, int number, std::vector<arg_complex>& args);

    // def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
    explicit C2PSAImpl(int c1, int c2, int n = 1, float e = 0.5);

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    // int get_outchannels()  ;

    torch::Tensor forward(torch::Tensor x)  ;
private:
    int _c;

    void Init_Modules(int c1, int c2, int n = 1, float e = 0.5);

public:
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
        float e = 0.5f, int k1 = 3, int k2 = 5, int d2 = 1);
    torch::Tensor forward(torch::Tensor x);
public:
    DSConv cv1{ nullptr };
    DSConv cv2{ nullptr };
    bool _b_add;
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
        float e = 0.5, int k1 = 3, int k2 = 5, int d2 = 1);

    torch::Tensor forward(torch::Tensor x);
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
    FuseModuleImpl(int c_in, bool channel_adjust);
    torch::Tensor forward(std::vector<torch::Tensor> x);

public:
    torch::nn::AvgPool2d downsample {nullptr};
    torch::nn::Upsample upsample {nullptr};    
    Conv conv_out{ nullptr };
};
TORCH_MODULE(FuseModule);

class AdaHyperedgeGenImpl : public torch::nn::Module
{
public:
    explicit AdaHyperedgeGenImpl(int node_dim, int num_hyperedges,
        int num_heads = 4, float dropout = 0.1,
        std::string context = "both");
    torch::Tensor forward(torch::Tensor x);

protected:
    int _num_hyperedges;
    int _num_heads;
    std::string _context;
    int _head_dim;
    double _scaling;

public:
    torch::Tensor prototype_base;
    torch::nn::Linear context_net{ nullptr };
    torch::nn::Linear pre_head_proj{ nullptr };
    torch::nn::Dropout dropout_net{ nullptr };
};
TORCH_MODULE(AdaHyperedgeGen);

class AdaHGConvImpl : public torch::nn::Module 
{
public:
    // def __init__(self, embed_dim, num_hyperedges=16, num_heads=4, 
    // dropout=0.1, context="both"):
    AdaHGConvImpl(int embed_dim, int num_hyperedge = 16, int num_heads = 4,
        float dropout = 0.1, std::string context = "both");
    torch::Tensor forward(torch::Tensor x);

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
        float dropout = 0.1, std::string context = "both");
    torch::Tensor forward(torch::Tensor x);

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
    C3AHImpl(int c1, int c2, float e = 1.0, int num_hyperedges = 8, std::string context = "bost");
    torch::Tensor forward(torch::Tensor x);

public:
    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    Conv cv3{ nullptr };
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
    using Conf_Params = std::tuple<int, int, bool, bool, float, float, std::string>;
    Conf_Params get_defaults()
    {
        return std::make_tuple(0, 8, true, false, 0.5f, 1.0f, "both");
    }
    HyperACEImpl(int in_channels, int number, std::vector<arg_complex>& args);

    //def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context="both", channel_adjust=True):
    explicit HyperACEImpl(int c1, int c2, int n = 1, int num_hyperedges = 8, bool dsc3k = true, bool shortcut = false,
        float e1 = 0.5f, float e2 = 1.0f, std::string context = "both", bool channel_adjust = true);

    // int get_outchannels()  ;

    torch::Tensor forward(std::vector<torch::Tensor> x) ;

private:
    int _b_dsc3k;
    void Init(int c1, int c2, int n = 1, int num_hyperedges = 8, bool dsc3k = true, bool shortcut = false,
        float e1 = 0.5f, float e2 = 1.0f, std::string context = "both", bool channel_adjust = true);
public:
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
    DownsampleConvImpl(int in_channels, int number, std::vector<arg_complex>& args);

    // def __init__(self, in_channels, channel_adjust=True):
    explicit DownsampleConvImpl(int in_channels, bool channel_adjust = true);

    // int get_outchannels()  ;

    torch::Tensor forward(torch::Tensor x) ;

private:
    bool _b_channel_adjust;
    void Init(int in_channels, bool channel_adjust = true);

public:
    torch::nn::AvgPool2d downsample{ nullptr };
    Conv cv_channel_adjust{ nullptr };
};
TORCH_MODULE(DownsampleConv);

class FullPAD_TunnelImpl : public BaseModule
{
public:
    FullPAD_TunnelImpl(int in_channels, int number, std::vector<arg_complex>& args);
    // def __init__(self, in_channels):
    explicit FullPAD_TunnelImpl(int in_channels);

    // int get_outchannels()  ;

    torch::Tensor forward(std::vector<torch::Tensor> x);
private:
    void Init(int in_channels);

public:
    torch::Tensor gate;
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
    using Conf_Params = std::tuple<int, bool, float, int , bool, int, int , int>;
    Conf_Params get_defaults()
    {
        return std::make_tuple(0, false, 0.5f, 1, true, 3, 7, 1);
    }
    DSC3k2Impl(int in_channels, int number, std::vector<arg_complex>& args);
//      def __init__(self, c1, c2, n = 1, dsc3k = False, 
//          e = 0.5, g = 1, shortcut = True, k1 = 3, k2 = 7,  d2 = 1 ):
    explicit DSC3k2Impl(int c1, int c2, int n = 1, bool dsc3k = false, float e = 0.5,
        int g = 1, bool shortcut = true, int k1 = 3, int k2 = 7, int d2 = 1);

    torch::Tensor forward(std::vector<torch::Tensor> x) {
        LOG(ERROR) << "DSC3k2Impl forward not support vector input";
        return x[0];
    }
    torch::Tensor forward(torch::Tensor x);

    // int get_outchannels() {return _out_channels;}
private:
    bool _b_dsc3k;
    Conf_Params def_params = get_defaults();

    void Init(int c1, int c2, int n = 1, bool dsc3k = false, float e = 0.5,
        int g = 1, bool shortcut = true, int k1 = 3, int k2 = 7, int d2 = 1);

public:

    Conv cv1{ nullptr };
    Conv cv2{ nullptr };
    torch::nn::ModuleList m;
};
TORCH_MODULE(DSC3k2);