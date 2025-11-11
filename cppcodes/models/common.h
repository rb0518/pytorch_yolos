#pragma once
// 对应着common.py文件中的定义
#include <BaseModel.h>
#include <torch/torch.h>

#include "conv.h"


class nnUpsampleImpl : public BaseModule
{
public:
    nnUpsampleImpl(){};
    nnUpsampleImpl(int in_channels, int number, std::vector<arg_complex>& args);

    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(torch::Tensor x) ;    
    int get_outchannels()  {return out_ch;}
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

class ContractImpl : public BaseModule
{
public:
    ContractImpl(){};
    ContractImpl(int in_channels, int number, std::vector<arg_complex>& args)
    {
        set_params(in_channels, number, args);
    }
    ~ContractImpl(){};
    
    void set_params(int in_channels, int number, std::vector<arg_complex>& args);

    torch::Tensor forward(torch::Tensor x) ;    
    int get_outchannels()  {return out_ch;}

    int in_ch;
    int out_ch;
    int gain;
};
TORCH_MODULE(Contract);

class ExpandImpl : public BaseModule
{
public:
    ExpandImpl(){};
    ~ExpandImpl(){};
    
    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(torch::Tensor x) ;    
    int get_outchannels()  {return out_ch;}

    int in_ch;
    int out_ch;
    int gain;
};
TORCH_MODULE(Expand);

#if 0
/*
*  libtorch中没有GELU，可以直接计算，为保持代码与python的一致，写成类
*   找到了torch::nn::GELU了
*/
class GELUImpl : public torch::nn::Module
{
public:
    GELUImpl() {};
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(GELU);
#endif

class InplaceSiLUImpl : public  torch::nn::Cloneable<InplaceSiLUImpl>
{
public:
    InplaceSiLUImpl(){};
    torch::Tensor forward(torch::Tensor input)
    {
        input = input * torch::sigmoid(input);
        return input;
    }
      void reset() override{};
    /// Pretty prints the `SiLU` module into the given `stream`.
    void pretty_print(std::ostream& stream) const override
    {
        stream << "torch::nn::Conv" << " (inplace == True) ";
    }
};
TORCH_MODULE(InplaceSiLU);

