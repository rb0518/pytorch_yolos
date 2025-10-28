#pragma once
#include <torch/torch.h>
#include <string>
#include <variant>

// network model 
// 所有的模型都继承自这个类，后期好进行扩展
class BaseModel : public torch::nn::Module
{
public:
public:
    virtual std::string getname() = 0;
    //virtual torch::Tensor forward(torch::Tensor x) = 0;

    // 2025-7-31 因为train返回的是3个tensor, 而predict返回的是tuple，c++不能像python一样灵活，
    // 统一采用std::vector<torch::Tensor> 返回结果，由外部判定
    virtual std::vector<torch::Tensor> forward(torch::Tensor x) = 0;
    
    explicit BaseModel(const std::string& yaml_file, int classes, int imagewidth, int imageheight, int channels, bool showdebuginfo)
    {
    };
    ~BaseModel(){};
    BaseModel(){};
};


// args 根据is_num决定是用n or s 
// s: 要根据情况决定，包含了None True False nearest nc, anchors等
using arg_complex = std::variant<int, std::string, bool, float>;

class BaseModule : public torch::nn::Module, std::enable_shared_from_this<BaseModule>
{
public:
    BaseModule(){};
    ~BaseModule(){};

    virtual torch::Tensor forward(std::vector<torch::Tensor> x) = 0;
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual int get_outchannels() = 0;   
    virtual void set_params(int inchannels, int number, std::vector<arg_complex>& args) = 0; 
};
