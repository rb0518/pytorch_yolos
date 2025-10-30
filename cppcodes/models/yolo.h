#pragma once

#include <string>
#include <vector>
#include <deque>
#include <tuple>
#include "BaseModel.h"
#include "yaml_load.h"

#include "common.h"
#include "conv.h"
#include "block.h"
#include "head.h"

class ModelImpl : public torch::nn::Module
{
public:
    int n_channels;                     //输入图像文件的channels
    int n_classes;                      //class类总数，这个不必要从yaml中读取
    std::vector<std::string> names;     // 类别名称

    int image_width;                    //输入图像的大小
    int image_height;
    bool b_showdebug;                   //控制调试信息的显示 

    std::string cfgfile;
    float gr = 1.0f;
   
    std::vector<std::vector<float>> anchors;
    std::deque<YoloLayerDef> layer_cfgs;
    // 记录下所有输入条件
    std::vector<std::vector<int>> layer_froms;
    std::vector<int> layer_out_chs;     // 记录每层out_channels

    std::vector<std::shared_ptr<BaseModule>> module_layers;

    bool is_segment = false;

    std::shared_ptr<DetectImpl> last_module;

    torch::Tensor stride;
    int get_stride_max(){ return stride.max().item().toInt();};   //???

    // 2025-10-17 增加从yolo.yaml文件名解释定义
    int yolo_version;
    std::string scale_id;
    std::string task;

    float depth_multiple;
    float width_multiple;
    int n_maxchannels = 512;

    bool legacy = true;

    // 调试开关
    VariantConfigs* p_args;
    void set_args_prt(VariantConfigs* _args){
        p_args = _args;
    }

public:
    ModelImpl(const std::string& yaml_file, int classes, int imagewidth, int imageheight, int channels, bool showdebuginfo = false);
    
    std::tuple<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor> forward(torch::Tensor x);
    
    void show_modelinfo();

    // 重写_apply方法处理子模块，对应到python中的_apply(self, fn)
//    torch::nn::Module& apply(const std::function<void(torch::nn::Module&)>& fn);
private:
    // 从yaml文件中读取network model的设置
    void readconfigs(const std::string& yaml_file, std::string scale_id = "n");
    // 根据参数，生成各层module
    void create_modules();
    void initialize_weights();
};
TORCH_MODULE(Model);
