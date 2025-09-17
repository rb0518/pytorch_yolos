#pragma once

#include <string>
#include <vector>
#include <deque>
#include <tuple>
#include "BaseModel.h"
#include "yaml_load.h"

// 这是按照YOLO air中Detect代码转换过来的
class DetectImpl : public torch::nn::Module {
public:
    DetectImpl(int _nc, std::vector<std::vector<float>> _anchors, std::vector<int> _ch, bool _inplace=true);
    
    // 因为train和predict的返回是不一样的， 所以还是建议将后续处理放到外部，网上的predict代码需要修改
    //std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);
    std::tuple<torch::Tensor, std::vector<torch::Tensor>> forward(std::vector<torch::Tensor> x);

private:
    std::tuple<torch::Tensor, torch::Tensor> _make_grid(int nx = 20, int ny = 20, int i = 0);

public:
    int nc;             // 类别数
    int no;             // 每个anchor的输出维度 = nc + 5
    int nl;             // 检测层数            [P3, P4, P5] = 3
    int na;             // 每个层的anchor数
    bool inplace;       // 是否启用原地操作
    //torch::Tensor anchors;

    int image_width = 640;
    int image_height = 640;

    torch::Tensor stride;
    std::vector<torch::Tensor> grid;
    std::vector<torch::Tensor> anchor_grid;
    std::vector<float> flat_anchors;
    torch::Tensor anchors_;
    //torch::Tensor anchor_grid;

    //torch::nn::ModuleList m;  //为了控制变量名，不采用ModuleList
    std::vector<torch::nn::Conv2d> m;
    void check_anchor_order();
    void _initialize_biases(std::vector<int> cf={});
};
TORCH_MODULE(Detect);

class ModelImpl : public torch::nn::Module
{
public:
    int n_channels;     //输入图像文件的channels
    int n_classes;      //class类总数，这个不必要从yaml中读取
    std::vector<std::string> names;     // 类别名称

    int image_width;    //输入图像的大小
    int image_height;
    bool b_showdebug;   //控制调试信息的显示 

    float depth_multiple;
    float width_multiple;

    std::string cfgfile;
    VariantConfigs hyp;
    float gr = 1.0f;
   
    std::vector<std::vector<float>> anchors;
    std::deque<YoloLayerDef> layer_cfgs;
    // 记录下所有输入条件
    std::vector<std::vector<int>> layer_froms;
    std::vector<int> layer_out_chs;     // 记录每层out_channels

    std::vector<std::shared_ptr<BaseModule>> module_layers;
    Detect module_detect{nullptr};
    torch::Tensor stride;
    int get_stride_max(){ return stride.max().item().toInt();};   //???

public:
    ModelImpl(const std::string& yaml_file, int classes, int imagewidth, int imageheight, int channels, bool showdebuginfo);
    
    //virtual torch::Tensor forward(torch::Tensor x) override;
    std::tuple<torch::Tensor, std::vector<torch::Tensor>> forward(torch::Tensor x);
    
    void show_modelinfo();

    // 重写_apply方法处理子模块，对应到python中的_apply(self, fn)
//    torch::nn::Module& apply(const std::function<void(torch::nn::Module&)>& fn);
private:
    // 从yaml文件中读取network model的设置
    void readconfigs(const std::string& yaml_file);
    // 根据参数，生成各层module
    void create_modules();
    void initialize_weights();
};
TORCH_MODULE(Model);
