#pragma once
#include <torch/torch.h>
#include <string>
#include <variant>

#include "globaldef.h"

// network model 
// 所有的模型都继承自这个类，后期好进行扩展
class BaseModelImpl : public torch::nn::Module
{
public:
    BaseModelImpl(std::string model_file, int channels, VariantConfigs* args);

    void init_model();

    void set_args_prt(VariantConfigs* _args);
    void set_classes(std::vector<std::string>& cls_names);
    void show_modelinfo(bool show_net = false);

    std::string get_taskname() { return this->task_name; }
    std::string get_scale_id() { return this->scale_id; }
    int get_imgsz() { return this->imgsz; }
    int get_modelversion() { return model_version; }

    virtual void load_modelsets() = 0;  // 从指定的model_file中读取model相关设置，可以是yaml文件，或者是.pt文件
    virtual void create_modelsets() = 0;
    virtual void init_weights() = 0;

    virtual void load_weights(torch::serialize::InputArchive& in_archive) = 0;    //  只读取model weights from 
    virtual void save_model(torch::serialize::OutputArchive& out_archive) = 0;

    virtual std::tuple<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor> forward(torch::Tensor x) = 0;

    virtual torch::Tensor get_stride() = 0;
    virtual int get_regmax() = 0;
protected:
    std::vector<torch::Tensor>  get_layer_inputs(int layer_id);

public:
    std::string cfg_file;
    int n_channels;
    VariantConfigs* p_args;
    int n_classes;                      //class类总数，这个不必要从yaml中读取
    std::vector<std::string> names;     // 类别名称    std::vector<std::string> names;

    int imgsz = 640;
    int model_version;
    std::string scale_id = "n";
    std::string task_name = "detect";
protected:
    std::vector<torch::Tensor> layer_outputs;
    std::deque<YoloLayerDef> layer_cfgs;
};
TORCH_MODULE(BaseModel);

class BaseModule : public torch::nn::Module
{
public:
    virtual torch::Tensor forward(std::vector<torch::Tensor> x)
    {
        LOG(ERROR) << "this is template class function forward(vector).";
        return x[0];
    }
    virtual torch::Tensor forward(torch::Tensor x)
    {
        LOG(ERROR) << "this is template class function forward.";
        return x;
    }

    int get_outchannels() { return _out_channels; }
protected:
    int _out_channels = 0;    
};
