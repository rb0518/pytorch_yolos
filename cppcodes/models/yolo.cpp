#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "yolo.h"
#include "yaml_load.h"
#include "common.h"
#include "utils.h"

#include "conv.h"
#include "block.h"

std::map<std::string, int> scales_key = {
        {"n" , 0},
        {"s" , 1},
        {"m" , 2},
        {"l" , 3},
        {"x" , 4}
    }; 

std::vector<std::string> base_modules = {
        "Classify", 
        "Conv",         
        "ConvTranspose", 
        "GhostConv", 
        "Bottleneck", 
        "GhostBottleneck", 
        "SPP", 
        "SPPF", 
        "C2fPSA", 
        "C2PSA", 
        "DWConv", 
        "Focus", 
        "BottleneckCSP", 
        "C1", 
        "C2", 
        "C2f", 
        "C3k2", 
        "RepNCSPELAN4", 
        "ELAN1", 
        "ADown", 
        "AConv", 
        "SPPELAN", 
        "C2fAttn", 
        "C3", 
        "C3TR", 
        "C3Ghost", 
        "torch.nn.ConvTranspose2d", 
        "DWConvTranspose2d", 
        "C3x", 
        "RepC3", 
        "PSA", 
        "SCDown", 
        "C2fCIB", 
        "A2C2f"
    };

bool isInVector(const std::vector<std::string>& vec, const std::string& str) 
{
    for (const auto& s : vec) {
        if (s == str) {
            return true;
        }
    }
    return false;
}        

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
        factories["C3"] = []() { return std::make_shared<C3Impl>(); };
        factories["Concat"] = []() { return std::make_shared<ConcatImpl>(); };
        factories["SPPF"] = []() { return std::make_shared<SPPFImpl>(); };
        factories["SPP"] = [](){ return std::make_shared<SPPImpl>(); };       
        factories["nn.Upsample"] = []() { return std::make_shared<nnUpsampleImpl>(); };
        factories["Focus"] = [](){ return std::make_shared<FocusImpl>(); };
        factories["Contract"] = []() { return std::make_shared<ContractImpl>(); };
        factories["Expand"] = []() { return std::make_shared<ExpandImpl>(); };
        factories["C3k2"] = []() { return std::make_shared<C3k2Impl>(); };
        factories["A2C2f"] = []() { return std::make_shared<A2C2fImpl>(); };
        factories["C2PSA"] = []() { return std::make_shared<C2PSAImpl>(); };
    }

    auto it = factories.find(className);
    if (it != factories.end()) {
        return it->second();
    } else {
        throw std::invalid_argument("Unknown class name: " + className);
    }
}

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
    std::string new_yamlfile = yaml_file;
    auto guess_yaml_task = [](std::string& filepath) {
        // 如给定为/path/to/yolov12n.yaml，但实际目录下只有yolov12.yaml, 人为指定scales="n"
            std::filesystem::path fs_fullpath = std::filesystem::path(filepath);
            bool fileexists_flag = std::filesystem::exists(fs_fullpath);
            auto [is_right, v, s, t] = parse_yolo_config(fs_fullpath.filename().string());
            if (false == is_right)
            {
                LOG(ERROR) << "cfg filename can't guess yolo version etc." << fs_fullpath.filename().string();
                return std::make_tuple(is_right, v, s, t, filepath);
            }

            if (s == "") 
            {
                s = "n";   // 默认取"n"
            }
            else {
                if (!fileexists_flag)
                {
                    std::string stem_name = fs_fullpath.stem().string();
                    size_t pos = stem_name.find(s);
                    if(pos != std::string::npos) 
                        stem_name.erase(pos, s.length());
                    filepath = fs_fullpath.parent_path().append(stem_name + fs_fullpath.extension().string()).string();
                }
            }
            return std::make_tuple(is_right, v, s, t, filepath);
        };

    auto [is_right, v, s, t, new_file] = guess_yaml_task(new_yamlfile);
    if(false == is_right) 
    {
        LOG(ERROR) << "can't guess the cfg filename. "; 
        exit(-1);
    }
    yolo_version = v;
    scale_id = s;
    task = t;

    std::cout << "guess cfgs: " << (is_right ? "true" : "false") << " v " << v << " sacle: " << s << " task: "
        << (t.empty() ? "Detect" : t) << std::endl;

    std::cout << "old path: " << yaml_file << "\n";
    std::cout << "exits path: " << new_file << "\n";

    readconfigs(new_file);
    create_modules();

    this->train();
    torch::Tensor img_tmp = torch::zeros({ 1, channels, imageheight, imagewidth });
    img_tmp = img_tmp.to(this->named_parameters().begin()->value().device());

    // std::cout << "create stride , input image: " << img_tmp.sizes() << std::endl;

    torch::Tensor pred;
    std::vector<torch::Tensor> train_ret(3);
    torch::Tensor pred_seg;
    std::tie(pred, train_ret, pred_seg) = forward(img_tmp);
    std::vector<int> strides;
    for(int i = 0; i < train_ret.size(); i++) {       // [bs, na, h, w, no]
        strides.emplace_back(image_height / train_ret[i].size(2));
    }

    last_module->stride = torch::tensor(strides);
    std::cout << "last_module->stride: " << last_module->stride << "\n";    // [8, 16, 32] CPULongType{3}
    //last_module->check_anchor_order();
    //std::cout << "before div: " << module_detect->anchors_ << std::endl;

    //last_module->anchors_ = last_module->anchors_.div(last_module->stride.view({ -1, 1, 1 }));

    //std::cout << "after div: " << module_detect->anchors_ << std::endl;

    stride = last_module->stride;
    last_module->bias_init(); 
    std::cout << "bias_init: over" << std::endl;
    initialize_weights();
    std::cout << "initialize_weights: over" << std::endl;
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
        //std::cout << "layer_id: "<< std::setw(5) << i << std::setw(20) << layer_cfgs[i].name << " " << layer_outputs[i].sizes() << std::endl;
    }


    // Detect & Segment convs forward
    std::vector<torch::Tensor> tmp;
    int module_id = layer_cfgs.size() - 1;    

    #ifdef _DEBUG_FOR_EXPORT_IMPORT_TENSORS_
    if(layer_outputs[0].size(0) == std::get<int>(p_args->at("batch")) && std::get<bool>(p_args->at("use_unified_batch")))
    {
        std::vector<torch::Tensor> new_layer_outputs;
        for(int i = 0; i < layer_outputs.size(); i++)
        {
            std::string layer_import_name = "ly" + std::to_string(i)+".pt";
            auto import_tensor = load_tensordata_from_file(layer_import_name);
            std::cout << i << " import_tensor: " << import_tensor.sizes() << " dtype: " << import_tensor.dtype() << " device: " << import_tensor.device().str() << std::endl;
            std::cout << i << " layer_outputs: " << layer_outputs[i].sizes() << " dtype: " << layer_outputs[i].dtype() << " device: " << layer_outputs[i].device().str() << std::endl;
            import_tensor = import_tensor.to(layer_outputs[i].dtype());
            import_tensor = import_tensor.to(layer_outputs[i].device());
            new_layer_outputs.push_back(import_tensor);
        }

        for(auto item : layer_froms[module_id])
        {
            tmp.push_back(new_layer_outputs[item]);
        }
    }
    else
    {
        for(auto item : layer_froms[module_id])
        {
            tmp.push_back(layer_outputs[item]);
        }
    }
    #else
    for(auto item : layer_froms[module_id])
    {
        tmp.push_back(layer_outputs[item]);
    }
    #endif

    if(is_segment == false)
    {
        std::tie(detect_pred, detect_outs) = last_module->forward(tmp);
        segment_pred = torch::empty({0}).to(detect_pred.device());
        //std::cout << "Detect return : " << detect_pred.sizes() << " proto: " << segment_pred.sizes() << std::endl;
    }
    else
    {
        // std::shared_ptr<SegmentImpl> last_s = std::dynamic_pointer_cast<SegmentImpl>(last_module);
        // std::tie(detect_pred, detect_outs, segment_pred) = last_s->forward(tmp);
        //std::cout << "Segment return : " << detect_pred.sizes() << " proto: " << segment_pred.sizes() << std::endl;
    }

    return { detect_pred, detect_outs, segment_pred};
}

void ModelImpl::readconfigs(const std::string& yaml_file, std::string scale_id /*= "n"*/)
{
    YAML::Node cfgs = YAML::LoadFile(yaml_file);
   
    int nc = cfgs["nc"].as<int>();
    if(nc != n_classes){
        LOG(WARNING) << "read nc " << nc << " no equil n_class. check cfg yaml.";
    }

    if (cfgs["depth_multiple"].IsDefined())
    {   // 旧格式文件中是放在一块的
        depth_multiple = cfgs["depth_multiple"].as<float>();
        width_multiple = cfgs["width_multiple"].as<float>();
    }
    else if(cfgs["scales"].IsDefined())
    {
        std::cout << "use new stype cfg yaml. \n";
        YAML::Node node_scales = cfgs["scales"];
        if (node_scales[scale_id].IsDefined())
        {
            std::vector<float> n_scale = node_scales[scale_id].as<std::vector<float>>();
            std::cout << "scale: " << scale_id <<" - depth: " << n_scale[0] << " width: " << n_scale[1] << " max_channels: " << int(n_scale[2]) << std::endl;
            depth_multiple = n_scale[0];
            width_multiple = n_scale[1];
            n_maxchannels = int(n_scale[2]);
        }
    }
    else
    {
        std::cout << "not found depth & width multiple and not found scales Node. \n";
    }


    // load anchors
    if (cfgs["anchors"].IsDefined())
    {
        YAML::Node node_anchors = cfgs["anchors"];
        if (!node_anchors.IsNull() && node_anchors.IsSequence())
        {
            for (YAML::const_iterator it = node_anchors.begin(); it != node_anchors.end(); ++it) {
                if ((*it).IsSequence()) {
                    anchors.push_back(it->as<std::vector<float>>());
                }
                else
                    LOG(ERROR) << "The yaml anchors define not suppert.";
            }
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
        //std::cout << "\n create layer " << i << " " << layer_cfgs[i].name;
        int ch_in = i == 0 ? this->n_channels : layer_out_chs[i - 1];
        layer_froms.push_back(layer_cfgs[i].froms);
        int n = calculate_depth_gain(layer_cfgs[i].number, depth_multiple);

        if(isInVector(base_modules, layer_cfgs[i].name))
        {
            int c2 = std::get<int>(layer_cfgs[i].args[0]);
            if(c2 != n_classes)
            {
                //std::cout <<" layer " << i << " name " << layer_cfgs[i].name << " old c2: " << c2;
                c2 = make_divisible(float(std::min(c2, n_maxchannels)) * width_multiple, 8);
                //std::cout <<" change to: " << c2 << "\n";
            }
            layer_cfgs[i].args[0] = c2;

            if(layer_cfgs[i].name == "C3k2")
            {
                legacy = false;
                if(scales_key[scale_id] >= scales_key["m"])
                {
                    std::cout << "C3k2: " << scale_id << " args[1] " << std::get<bool>(layer_cfgs[i].args[1]);
                    layer_cfgs[i].args[1] = true;
                    std::cout << " change to : " << std::get<bool>(layer_cfgs[i].args[1]) << "\n";
                }
            }
            if(layer_cfgs[i].name == "A2C2f")
            {
                legacy = false;
                /*
                    if scale in "lx":  # for L/X sizes
                        args.extend((True, 1.2))
                */
                if(scales_key[scale_id] >= scales_key["l"])
                {
                    std::cout << "A2C2f: " << scale_id << " add args [3] & [4] ";                    
                    layer_cfgs[i].args[3] = true;
                    layer_cfgs[i].args[4] = 1.2f;
                }
            }
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
        last_module = std::make_shared<DetectImpl>(n_classes, inchannels);
    }
    /*
    else{
        is_segment = true;
        std::cout << "Create segment layer: " << std::endl;
        int nm = std::get<int>(final_layercfgs.args[2]);
        int npr = std::get<int>(final_layercfgs.args[3]);
            //c2 = make_divisible(float(c2) * width_multiple, 8);
        int tmp_npr = npr;
        std::cout << "Segment " << "in " << tmp_npr;
        tmp_npr = make_divisible(float(tmp_npr) * width_multiple, 8);
        std::cout << " after : " << tmp_npr << std::endl;
        last_module = std::make_shared<SegmentImpl>(n_classes, anchors, nm, tmp_npr, inchannels, false);
    }
    */
    register_module("model-" + std::to_string(layer_cfgs.size()-1), last_module);    
    //std::cout << "Create modules over..." << std::endl;
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