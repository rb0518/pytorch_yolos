#pragma once

#include <string>
#include <vector>
#include <deque>
#include <tuple>

#include "BaseModel.h"
#include "head.h"

#include "globaldef.h"


class ModelImpl : public BaseModelImpl
{
public:
    ModelImpl(std::string model_file, int channels, VariantConfigs* args);

    virtual void load_modelsets();  // 从指定的model_file中读取model相关设置，可以是yaml文件，或者是.pt文件
    virtual void create_modelsets();
    virtual void init_weights();

    virtual void load_weights(torch::serialize::InputArchive& in_archive);    //  只读取model weights from 
    virtual void save_model(torch::serialize::OutputArchive& out_archive);

    virtual std::tuple<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor> forward(torch::Tensor x);

    virtual torch::Tensor get_stride();
    virtual int get_regmax();

protected:
    void load_model_from_yaml();
    std::tuple<bool, int, std::string> load_model_from_ptfile();    
    void parse_model_sets();    // 从cfg_buffer内存中析构网络设置

public:
    bool b_showdebug = false;                   //控制调试信息的显示 
    float gr = 1.0f;
    bool is_segment = false;

    std::vector<std::vector<float>> anchors;
    torch::Tensor stride;
    int get_stride_max(){ return stride.max().item().toInt();};   //???

    // 2025-10-17 增加从yolo.yaml文件名解释定义
    float depth_multiple;
    float width_multiple;
    int n_maxchannels = 512;

    bool legacy = true;

    std::vector<torch::Tensor> detect_outs;

    // 2025-11-8 测试用ModuleList存储网络，不像python代码一样用Sequential的原因是Sequential在libtorch中
    // 是必须要明确压入的Module，包括这些Module中的都需要是统一的forward输入与输出
    torch::nn::ModuleList _modules;

    std::vector<unsigned char> cfg_buffer;

public:
    std::shared_ptr<DetectImpl> getLastModulePtr();

protected:
    // 这两个函数如果后续需要，可以变更为基类函数或虚函数
    bool readModelCfgFile(std::string yaml_file);
    bool readCfgFromPtFile(std::string pt_file);

    void create_modules();      // 根据参数，生成各层module
    void initialize_weights();
};
TORCH_MODULE(Model);
