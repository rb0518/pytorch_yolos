#pragma once
// 对应着common.py文件中的定义
#include <BaseModel.h>
#include <torch/torch.h>

class ConvImpl : public BaseModule
{
    //c1, c2, k=1, s=1, p=None, g=1, act=True
public:
    ConvImpl(){};
    ~ConvImpl(){};
    explicit ConvImpl(int c1, int c2, int k, int s, int p);
    
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

    void Init_Modules(int c1, int c2, int k, int s = 1, int p = -1);
//    torch::nn::Sequential conv;
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };
    torch::nn::SiLU silu{ nullptr };

    torch::nn::Conv2d fusedconv{ nullptr };
};
TORCH_MODULE(Conv);


// # Standard bottleneck
// # ch_in, ch_out, shortcut, groups, expansion
class BottleneckImpl : public torch::nn::Module
{
public:
    explicit BottleneckImpl(int c1, int c2, bool shortcut, float e);
    // 暂时还不需要，只由C3调用
    torch::Tensor forward(std::vector<torch::Tensor> x);
    torch::Tensor forward(torch::Tensor x);
    int get_outchannels(){return out_ch;};
private:
    int in_ch;
    int out_ch;
    bool b_shortcut;
    float expansion;

    Conv cv1;
    Conv cv2;
};
TORCH_MODULE(Bottleneck);

/*
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
*/
class C3Impl : public BaseModule
{
public:
    explicit C3Impl(int c1, int c2, int n=1, bool shortcut=true, float e=0.5);
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
    float expansion;    // python code, bottleneck定义没用传入值，直接用了1.0

    void Init_Modules(int c1, int c2, int n=1, bool shortcut=true, float e=0.5);


    Conv cv1;
    Conv cv2;
    Conv cv3;
    // torch::nn::Sequential m; //不好自控命名规则
    std::vector<Bottleneck> bottlenecks;
};
TORCH_MODULE(C3);


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


class nnUpsampleImpl : public BaseModule
{
public:
    nnUpsampleImpl(){};
    ~nnUpsampleImpl(){};

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;    
    int get_outchannels() override {return out_ch;}
public:
    int in_ch;
    int out_ch;
    int64_t scale_size;
    std::string str_modetype;
#if 0
    bool bneed_calcsize = false;
    torch::nn::Upsample upsample{ nullptr };
#endif
};
TORCH_MODULE(nnUpsample);


class SPPFImpl : public BaseModule
{
public:
    SPPFImpl(){};
    ~SPPFImpl(){};

    void Init_Modules(int c1, int c2, int k);

    void  set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x)override;
    torch::Tensor forward(torch::Tensor x)override; 
    int get_outchannels() override {return out_ch;} ;    
public:
    int in_ch;
    int out_ch;
    int k;
    Conv cv1;
    Conv cv2;
    torch::nn::MaxPool2d m{nullptr};
};
TORCH_MODULE(SPPF);

class SPPImpl : public BaseModule
{
public:
    SPPImpl(){};
    ~SPPImpl(){};

    void Init_Modules(int c1, int c2);

    void  set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x)override;
    torch::Tensor forward(torch::Tensor x)override; 
    int get_outchannels() override {return out_ch;} ;    
public:
    int in_ch;
    int out_ch;
    std::vector<int> k = {5, 9, 13};
    Conv cv1;
    Conv cv2;
    torch::nn::ModuleList m;
};
TORCH_MODULE(SPP);

class FocusImpl : public BaseModule
{
public:
    FocusImpl(){};
    ~FocusImpl(){};

    explicit FocusImpl(int c1, int c2, int k, int s, int p);
    
    void set_params(int inchannels, int number, std::vector<arg_complex>& args);
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

    void Init_Modules(int c1, int c2, int k, int s = 1, int p = -1);    
};
TORCH_MODULE(Focus);


