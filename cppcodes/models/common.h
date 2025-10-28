#pragma once
// 对应着common.py文件中的定义
#include <BaseModel.h>
#include <torch/torch.h>

#include "conv.h"


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

class ContractImpl : public BaseModule
{
public:
    ContractImpl(){};
    ~ContractImpl(){};
    
    void set_params(int in_channels, int number, std::vector<arg_complex>& args);
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;    
    int get_outchannels() override {return out_ch;}

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
    torch::Tensor forward(std::vector<torch::Tensor> x) override;
    torch::Tensor forward(torch::Tensor x) override;    
    int get_outchannels() override {return out_ch;}

    int in_ch;
    int out_ch;
    int gain;
};
TORCH_MODULE(Expand);


