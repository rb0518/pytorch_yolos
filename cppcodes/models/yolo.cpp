#include <yaml-cpp/yaml.h>

#include "yolo.h"
#include "yaml_load.h"
#include "common.h"
#include "utils.h"

int calculate_depth_gain(int n, float gd) {
    return (n > 1) ? std::max(static_cast<int>(std::round(n * gd)), 1) : n;
}

int make_divisible(int x, int divisor) {
    return static_cast<int>(std::ceil(x / static_cast<double>(divisor))) * divisor;
}

template<typename T, typename... Args>
std::shared_ptr<BaseModule> createModule(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

std::shared_ptr<BaseModule> createObject(const std::string& className) {
    static std::map<std::string, std::function<std::shared_ptr<BaseModule>()>> factories;

    // 初始化工厂映射表（这里仅作为示例）
    if (factories.empty()) {
        factories["Conv"] = []() { return std::make_shared<ConvImpl>(); };
        factories["BottleneckCS"] = [](){ return std::make_shared<BottleneckCSPImpl>();};
        factories["C3"] = []() { return std::make_shared<C3Impl>(); };
        factories["Concat"] = []() { return std::make_shared<ConcatImpl>(); };
        factories["SPPF"] = []() { return std::make_shared<SPPFImpl>(); };
        factories["SPP"] = [](){ return std::make_shared<SPPImpl>(); };       
        factories["nn.Upsample"] = []() { return std::make_shared<nnUpsampleImpl>(); };
        factories["Focus"] = [](){ return std::make_shared<FocusImpl>(); };
        factories["Contract"] = []() { return std::make_shared<ContractImpl>(); };
        factories["Expand"] = []() { return std::make_shared<ExpandImpl>(); };
    }

    auto it = factories.find(className);
    if (it != factories.end()) {
        return it->second();
    } else {
        throw std::invalid_argument("Unknown class name: " + className);
    }
}

// ===================== Detect module start ===========================
DetectImpl::DetectImpl(int _nc, std::vector<std::vector<float>> _anchors, std::vector<int> _ch, 
                    bool _inplace, bool _is_segment) 
{
    this->nc = _nc;
    this->no = _nc + 5;
    this->nl = _anchors.size();          // float
    this->na = _anchors[0].size() / 2;
    this->inplace = _inplace;
    this->is_segment = _is_segment;

    for (int i = 0; i < nl; i++)
    {
        this->grid.push_back(torch::zeros({1}));
        this->anchor_grid.emplace_back(torch::zeros({1}));
    }

    for(int i = 0; i < _anchors.size(); i++)
    {
        for(int j = 0; j < _anchors[i].size(); j++)
            flat_anchors.push_back(_anchors[i][j]);
    }
    anchors_ = torch::tensor(flat_anchors).view({this->nl, -1, 2});
    register_buffer("anchors", anchors_);

    if(false == is_segment)
    {
        for(int i = 0; i < _ch.size(); i++)
            m->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(_ch[i], this->no * this->na, 1)));
        register_module("m", m);
    }
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> DetectImpl::forward(std::vector<torch::Tensor> x) 
{
    std::vector<torch::Tensor> ret(this->m->size()); // 用ModuleList，解决了named索引的一致，但后续操作不方便
    std::vector<torch::Tensor> z(this->m->size());
    torch::Tensor z_cat = torch::zeros({ 0 });
    for (int i = 0; i < this->nl; i++)
    {
        auto mi = m[i]->as<torch::nn::Conv2d>();
        ret[i] = mi->forward(x[i]);
        auto bs = ret[i].size(0);     // batch_size
        auto ny = ret[i].size(2);     // h
        auto nx = ret[i].size(3);     // w
    
        anchors_ = anchors_.to(x[i].device());

        // # x(bs,255,20,20) to x(bs,3,20,20,85)
        ret[i] = ret[i].view({bs, this->na, this->no, ny, nx}).permute({0, 1, 3, 4, 2}).contiguous();

        if (!is_training()) 
        {  // predict 需要解码数据转到bbox模式[xy, hw, conf, pred_cls]
            bool bneed_make_grid = true;
            if(bneed_make_grid)
            {
                std::tie(this->grid[i], this->anchor_grid[i]) = _make_grid(nx, ny, i);
            }    
            
            this->grid[i] = this->grid[i].to(x[i].device());
            this->stride = this->stride.to(x[i].device());
            this->anchor_grid[i] = this->anchor_grid[i].to(x[i].device());
            auto splits = ret[i].sigmoid().split({2, 2, nc + 1}, 4);
            auto xy = (splits[0] * 2 + grid[i]) * stride[i];
            auto wh = (splits[1] * 2).pow(2) * anchor_grid[i];
            auto y = torch::cat({xy, wh, splits[2]}, 4);   
            z[i] = y.view({bs, na* nx * ny, no}).clone();
        }
    }

    if(!is_training())
    {
        z_cat = torch::cat(z, 1);   //将z放到最前面 [bs, 25200, 85]
    }
    return { z_cat, ret};    // 统一返回格式，如是是predict return [x, z = []]
}

std::tuple<torch::Tensor, torch::Tensor> DetectImpl::_make_grid(int nx, int ny, int i)
{
    torch::Device d = this->named_buffers()["anchors"].device();
    auto yvxv = torch::meshgrid({ torch::arange(ny).to(d), torch::arange(nx).to(d)}, "ij");
    auto yv = yvxv[0];
    auto xv = yvxv[1];
    auto ret = torch::stack({ xv, yv}, 2).expand({1, na, ny, nx, 2}).to(torch::kFloat) - 0.5f;
    auto an_grid = (anchors_[i].clone() * this->stride[i]).view({ 1, this->na, 1, 1, 2 })
            .expand({ 1, this->na, ny, nx, 2 }).to(torch::kFloat);
    return { ret, an_grid };
}

void DetectImpl::check_anchor_order()
{
    auto da = flat_anchors[flat_anchors.size()-1] - flat_anchors[0];
    auto ds = stride[stride.size(0)-1].item().toFloat() - stride[0].item().toFloat();
    if ((da * ds) < 0)
    {
        LOG(ERROR) << "Reversing anchor order.";
        named_buffers()["anchors"] = named_buffers()["anchors"].flip(0);
    }
}

// Initialize biases
void DetectImpl::_initialize_biases(std::vector<int> cf)
{
    if(stride.size(0) != m->size())
    {
        LOG(ERROR) << "stride size not equil m size.";
        return;
    }

    for (size_t i = 0; i < m->size(); ++i) {
        auto s = stride[i].item().toFloat();
        auto mi = m[i]->as<torch::nn::Conv2d>();
        auto b = mi->bias.view({ na, -1 }); // conv.bias(255) to (3,85)
        b.data().index_put_({torch::indexing::Slice(), 4},
            b.index({ torch::indexing::Slice(), 4 }) +
            std::log(8.0f / std::pow((640.0f / s), 2))); // obj (8 objects per 640 image)


        b.data().index_put_({ torch::indexing::Slice(), torch::indexing::Slice(5, torch::indexing::None) },
            b.index({ torch::indexing::Slice(), torch::indexing::Slice(5, torch::indexing::None) }) +
            std::log(0.6f / (float(nc) - 0.99999f)));
 
        mi->bias.data().copy_(b.view({-1}));
        mi->bias.set_requires_grad(true);
    }
}
// ===================== Detect module over ===========================

// =====================  Segment Moduel ===========================
SegmentImpl::SegmentImpl(int _nc, std::vector<std::vector<float>> _anchors, 
                    int _nm, int _npr, std::vector<int> _ch, bool _inplace) 
                    : DetectImpl(_nc, _anchors, _ch, _inplace, true)
{
    this->nm = _nm;
    this->npr = _npr;
    this->no = _nc + 5 + nm;    // number of outputs per anchor
    for(int i = 0; i < _ch.size(); i++)
    {
        m->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(_ch[i], this->no * this->na, 1)));
    }
    register_module("m", m);
    proto = Proto(_ch[0], npr, nm);
    register_module("proto", proto);
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor> SegmentImpl::forward(std::vector<torch::Tensor> x) 
{
    std::vector<torch::Tensor> ret(this->m->size()); // 用ModuleList，解决了named索引的一致，但后续操作不方便
    std::vector<torch::Tensor> z(this->m->size());
    torch::Tensor z_cat = torch::zeros({ 0 });

    torch::Tensor p_proto = proto->forward(x[0]);

    for (int i = 0; i < this->nl; i++)
    {
        auto mi = m[i]->as<torch::nn::Conv2d>();
        ret[i] = mi->forward(x[i]);
        auto bs = ret[i].size(0);     // batch_size
        auto ny = ret[i].size(2);     // h
        auto nx = ret[i].size(3);     // w
    
        anchors_ = anchors_.to(x[i].device());

        // # x(bs,255,20,20) to x(bs,3,20,20,85)
        ret[i] = ret[i].view({bs, this->na, this->no, ny, nx}).permute({0, 1, 3, 4, 2}).contiguous();

        if (!is_training()) 
        {  // predict 需要解码数据转到bbox模式[xy, hw, conf, pred_cls]
            bool bneed_make_grid = true;
            if(bneed_make_grid)
            {
                std::tie(this->grid[i], this->anchor_grid[i]) = _make_grid(nx, ny, i);
            }    
            
            this->grid[i] = this->grid[i].to(x[i].device());
            this->stride = this->stride.to(x[i].device());
            this->anchor_grid[i] = this->anchor_grid[i].to(x[i].device());
            auto splits = ret[i].split_with_sizes({2, 2, nc + 1, no - nc - 5}, 4);
            auto xy = (splits[0].sigmoid() * 2 + grid[i]) * stride[i];
            auto wh = (splits[1].sigmoid() * 2).pow(2) * anchor_grid[i];
            auto y = torch::cat({xy, wh, splits[2].sigmoid(), splits[3]}, 4);   
            z[i] = y.view({bs, na* nx * ny, no}).clone();
        }
    }

    if(!is_training())
    {
        z_cat = torch::cat(z, 1);   //将z放到最前面 [bs, 25200, 85]
    }
    return { z_cat, ret, p_proto};    // 统一返回格式，如是是predict return [x, z = []]
}
// ===================== Segment module over ===========================
ModelImpl::ModelImpl(const std::string& yaml_file, int classes, int imagewidth, int imageheight, int channels, bool showdebuginfo)
{
    n_classes = classes;
    image_width = imagewidth;
    image_height = imageheight;
    n_channels = channels;
    b_showdebug = showdebuginfo;
    cfgfile = yaml_file;
    if(b_showdebug)
        std::cout << "Config YAML file: " << yaml_file << std::endl;

    readconfigs(yaml_file);
    create_modules();

    this->train();
    torch::Tensor img_tmp = torch::zeros({ 1, channels, imageheight, imagewidth });
    img_tmp = img_tmp.to(this->named_parameters().begin()->value().device());
    torch::Tensor pred;
    std::vector<torch::Tensor> train_ret(3);
    torch::Tensor pred_seg;
    std::tie(pred, train_ret, pred_seg) = forward(img_tmp);
    std::vector<int> strides;
    for(int i = 0; i < train_ret.size(); i++) {       // [bs, na, h, w, no]
        strides.emplace_back(image_height / train_ret[i].size(2));
    }

    last_module->stride = torch::tensor(strides);
    last_module->check_anchor_order();
    //std::cout << "before div: " << module_detect->anchors_ << std::endl;

    last_module->anchors_ = last_module->anchors_.div(last_module->stride.view({ -1, 1, 1 }));

    //std::cout << "after div: " << module_detect->anchors_ << std::endl;

    stride = last_module->stride;
    last_module->_initialize_biases(); 

    initialize_weights();
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor> ModelImpl::forward(torch::Tensor x)
{
    if(b_showdebug)
        std::cout << "Start forward() total layers: " << module_layers.size()  << std::endl;

    std::vector<torch::Tensor> layer_outputs;
    std::vector<torch::Tensor> detect_outs;
    torch::Tensor detect_pred;
    torch::Tensor segment_pred;

    for(int i = 0; i < module_layers.size(); i++)
    {
        //std::cout << "run layer: " << i << std::endl;
        if(i == 0)
        {
            layer_outputs.push_back(module_layers[0]->forward(x));
        }
        else
        {
            std::vector<torch::Tensor> tmp;
            for(auto item : layer_froms[i])
            {
                if(item < 0)    
                    tmp.push_back(layer_outputs[i+item]);
                else
                    tmp.push_back(layer_outputs[item]);
            }

            if(tmp.size() == 1)
                layer_outputs.push_back(module_layers[i]->forward(tmp[0]));
            else
                layer_outputs.push_back(module_layers[i]->forward(tmp));
        }
        if(b_showdebug)
            std::cout << "layer_id: "<< std::setw(5) << i << std::setw(20) << layer_cfgs[i].name << " " << layer_outputs[i].sizes() << std::endl;
    }

    // Detect & Segment convs forward
    int module_id = layer_cfgs.size() - 1;
    std::vector<torch::Tensor> tmp;
    for(auto item : layer_froms[module_id])
    {
        tmp.push_back(layer_outputs[item]);
    }

    if(is_segment == false)
    {
        std::tie(detect_pred, detect_outs) = last_module->forward(tmp);
        segment_pred = torch::empty({0});
    }
    else
    {
        std::shared_ptr<SegmentImpl> last_s = std::dynamic_pointer_cast<SegmentImpl>(last_module);
        std::tie(detect_pred, detect_outs, segment_pred) = last_s->forward(tmp);
    }

    return { detect_pred, detect_outs, segment_pred};
}

void ModelImpl::readconfigs(const std::string& yaml_file)
{
    YAML::Node cfgs = YAML::LoadFile(yaml_file);
   
    //std::cout << cfgs << std::endl;
    
    int nc = cfgs["nc"].as<int>();
    depth_multiple = cfgs["depth_multiple"].as<float>();
    width_multiple = cfgs["width_multiple"].as<float>();

    // load anchors
    YAML::Node node_anchors = cfgs["anchors"];   
    if(!node_anchors.IsNull() && node_anchors.IsSequence())
    {
        for(YAML::const_iterator it = node_anchors.begin(); it != node_anchors.end(); ++it){
            if((*it).IsSequence()){
                anchors.push_back(it->as<std::vector<float>>());
            }
            else
                LOG(ERROR) << "The yaml anchors define not suppert.";
        }
    }
   
    YAML::Node node_backbone = cfgs["backbone"];
    YAML::Node node_head = cfgs["head"];

    int total_layers = node_backbone.size() + node_head.size();

    auto read_one_layer =[&](YAML::Node& cfgs){
        YoloLayerDef layer;
        // froms 
        if(cfgs[0].IsSequence())
        {
            layer.froms = cfgs[0].as<std::vector<int>>();
        }
        else
            layer.froms.push_back(cfgs[0].as<int>());

        layer.number = cfgs[1].as<int>();
        layer.name = cfgs[2].as<std::string>();
        // if(layer.name == "SPP")
        //     std::cout << cfgs[3] << std::endl;
        for(auto e : cfgs[3])
        {
            if(e.IsSequence())  // 针对SPP中第二参数为数组，修正
            {
                std::stringstream temp;
                temp << e.as<std::vector<float>>();
                layer.add_one_arg(temp.str());
            }
            else
                layer.add_one_arg(e.as<std::string>());
        }
        return layer;
    };


    for(int i = 0; i < node_backbone.size(); i++)
    {
        YAML::Node layer = node_backbone[i];
        layer_cfgs.push_back(read_one_layer(layer));
    }

    for(int i = 0; i < node_head.size(); i++)
    {
        YAML::Node layer = node_head[i];
        layer_cfgs.push_back(read_one_layer(layer));    
    }
}

void ModelImpl::create_modules()
{
    /*
    std::vector<std::vector<int>> layer_froms;
    std::vector<int> layer_out_chs;     // 记录每层out_channels
    std::vector<std::shared_ptr<BaseModule>> modules;
    */
    module_layers.clear();
    module_layers.resize(layer_cfgs.size()-1);    // Detect Segment单独构造
    for (int i = 0; i < layer_cfgs.size() - 1; i++)
    {
        int ch_in = i == 0 ? this->n_channels : layer_out_chs[i - 1];
        layer_froms.push_back(layer_cfgs[i].froms);
        int n = calculate_depth_gain(layer_cfgs[i].number, depth_multiple);

        if (layer_cfgs[i].name == "Conv" ||
            layer_cfgs[i].name == "SPP" ||
            layer_cfgs[i].name == "SPPF" ||
            layer_cfgs[i].name == "C3" ||
            layer_cfgs[i].name == "Focus")
        {
            int c2 = std::get<int>(layer_cfgs[i].args[0]);
            //std::cout << "c2 - 0: " << c2 << " " << width_multiple;
            c2 = make_divisible(float(c2) * width_multiple, 8);
            //std::cout << " c2 - 1: " << c2 << std::endl;

            layer_cfgs[i].args[0] = c2;
        }
        // 根据layer_cfgs[i].name, 生成对应的Module，并压入vector中
        module_layers[i] = createObject(layer_cfgs[i].name);
        module_layers[i]->set_params(ch_in, 
                    n,
                    layer_cfgs[i].args
                    );

        register_module("model-"+std::to_string(i), module_layers[i]);
        int total_out_ch = module_layers[i]->get_outchannels();
        if(layer_cfgs[i].name == "Concat")       // 合并要将多个前面out_channel加起来算
        {
            total_out_ch = 0;
            for(auto j : layer_cfgs[i].froms)
            {
                if(j < 0)
                {
                    total_out_ch += layer_out_chs[(i+j)];
                }
                    
                else
                    total_out_ch += layer_out_chs[j];
            }
        }
        layer_out_chs.push_back(total_out_ch);            
        //std::cout << "Layer: " << i << " " << layer_cfgs[i].name << " channels in: " << ch_in << " out:" << layer_out_chs[i] << std::endl;
    }
    // 构造 最后一层的 Detect/Segment
    YoloLayerDef final_layercfgs = layer_cfgs[layer_cfgs.size()-1];
    layer_froms.push_back(final_layercfgs.froms);

    std::vector<int> inchannels;
    for(auto item : final_layercfgs.froms)
        inchannels.push_back(layer_out_chs[item]);

    if(final_layercfgs.name == "Detect") 
    {
        last_module = std::make_shared<DetectImpl>(n_classes, anchors, inchannels, false);
    }
    else{
        is_segment = true;
        std::cout << "Create segment layer: " << std::endl;
        int nm = std::get<int>(final_layercfgs.args[2]);
        int npr = std::get<int>(final_layercfgs.args[3]);

        last_module = std::make_shared<SegmentImpl>(n_classes, anchors, nm, npr, inchannels, false);
    }
    register_module("model-" + std::to_string(layer_cfgs.size()-1), last_module);    
}

void ModelImpl::initialize_weights()
{
    for (auto& submodule : this->modules(false))
    {
        if (submodule->name() == "torch::nn::Conv2dImpl") {
            // Kaiming初始化被注释掉
            // torch::nn::init::kaiming_uniform_(submodule->as<torch::nn::Conv2d>()->weight);
            // torch::nn::init::constant_(submodule->as<torch::nn::Conv2d>()->bias, 0);
            // torch::nn::init::kaiming_normal_(submodule->as<torch::nn::LeakyReLU>()->weight, 
            //     torch::nn::init::FanMode::FanOut, 
            //     torch::nn::init::Nonlinearity::ReLU);
        }
        else if (submodule->name() == "torch::nn::BatchNorm2dImpl") {
            //std::cout << "find module: " << submodule->name() << std::endl;
            submodule->as<torch::nn::BatchNorm2d>()->options.eps(1e-3);
            submodule->as<torch::nn::BatchNorm2d>()->options.momentum(0.03);
        }
        else if (submodule->name() == "torch::nn::LeakyReLUImpl") {
            submodule->as<torch::nn::LeakyReLU>()->options.inplace(true);
        }
        else if (auto act = submodule->as<torch::nn::ReLU>()) {
            submodule->as<torch::nn::ReLU>()->options.inplace(true);
        }
        else if (auto act = submodule->as<torch::nn::ReLU6>()) {
            submodule->as<torch::nn::ReLU6>()->options.inplace(true);
        }
    }
}

void ModelImpl::show_modelinfo()
{
    std::cout << ColorString("Model: ", "Info") << this->cfgfile << std::endl;
    std::cout << "nc: " << n_classes << " ch: " << n_channels << " height: " << image_height << " widht: " << image_width << std::endl;
    int i = 0;
    std::for_each(layer_cfgs.begin(),layer_cfgs.end(),[&](YoloLayerDef layer){
        std::cout << std::setw(3) << i << "  " << layer.show_info() << std::endl; i+=1;
        });  

    std::cout << "Total layers: " << layer_cfgs.size() << std::endl;
}