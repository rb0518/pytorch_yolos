#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "yolo.h"
#include "yaml_load.h"
#include "common.h"
#include "utils.h"

#include "conv.h"
#include "block.h"

#include "torch_utils.h"

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
        "A2C2f",

        "DSC3k2",       // V13
        "HyperACE",
        "DSConv"
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

// template<typename T, typename... Args>
// std::shared_ptr<BaseModule> createModule(Args&&... args) {
//     return std::make_shared<T>(std::forward<Args>(args)...);
// }

#if 1
#define REGISTER_LAYER_FACTORY(LAYER_NAME) \
    else if(className == #LAYER_NAME) \
    { \
        auto layer = LAYER_NAME(in_channels, number, args); \
        auto out_ch = layer->get_outchannels(); \
        modulelist->push_back(layer.ptr()); \
        return out_ch; \
    }    

int AppendLayer(torch::nn::ModuleList& modulelist, 
    const std::string& className, int in_channels, int number, std::vector<arg_complex> args)
{
    if (className == "nn.Upsample")
    {
        double scale_size = static_cast<double>(std::get<int>(args[1]));
        std::string str_modetype = std::get<std::string>(args[2]);
        if(str_modetype != "nearest")
            LOG(ERROR) << "program only support nearest: " << str_modetype;
        modulelist->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions().
            mode(torch::kNearest).
            scale_factor(std::vector<double>({ double(scale_size), double(scale_size) }))));
        return in_channels;
    }
    REGISTER_LAYER_FACTORY(Conv)
    REGISTER_LAYER_FACTORY(C3)
    REGISTER_LAYER_FACTORY(Concat)
    REGISTER_LAYER_FACTORY(SPPF)
    REGISTER_LAYER_FACTORY(SPP)
    REGISTER_LAYER_FACTORY(Focus)
    REGISTER_LAYER_FACTORY(Contract)
    REGISTER_LAYER_FACTORY(C3k2)
    REGISTER_LAYER_FACTORY(A2C2f)
    REGISTER_LAYER_FACTORY(C2PSA)
    // 2025-11-1 add for v13  
    REGISTER_LAYER_FACTORY(DSConv)
    REGISTER_LAYER_FACTORY(HyperACE)
    REGISTER_LAYER_FACTORY(DownsampleConv)
    REGISTER_LAYER_FACTORY(FullPAD_Tunnel)
    REGISTER_LAYER_FACTORY(DSC3k2)
    // 所有新添加的类放在这里
    else
        throw std::invalid_argument("Unknown class name: " + className);

    return 0;
}
#endif
#if 1
std::shared_ptr<BaseModule> createObject(const std::string& className, 
    int in_channels, int number, std::vector<arg_complex> args) {
    static std::map<std::string, std::function<std::shared_ptr<BaseModule>(int, int, std::vector<arg_complex>)>> factories;

    // 初始化工厂映射表（这里仅作为示例）
    if (factories.empty()) {
        factories["Conv"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<ConvImpl>(in_channels, number, args); };
        factories["C3"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<C3Impl>(in_channels, number, args); };
        factories["Concat"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<ConcatImpl>(in_channels, number, args); };
        factories["SPPF"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<SPPFImpl>(in_channels, number, args); };
        factories["SPP"] = [](int in_channels, int number, std::vector<arg_complex> args){ return std::make_shared<SPPImpl>(in_channels, number, args); };       
        factories["nn.Upsample"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<nnUpsampleImpl>(in_channels, number, args); };
        factories["Focus"] = [](int in_channels, int number, std::vector<arg_complex> args){ return std::make_shared<FocusImpl>(in_channels, number, args); };
        factories["Contract"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<ContractImpl>(in_channels, number, args); };
//        factories["Expand"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<ExpandImpl>(in_channels, number, args); };
        factories["C3k2"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<C3k2Impl>(in_channels, number, args); };
        factories["A2C2f"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<A2C2fImpl>(in_channels, number, args); };
        factories["C2PSA"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<C2PSAImpl>(in_channels, number, args); };

        // 2025-11-1 add for v13      
        factories["DSConv"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<DSConvImpl>(in_channels, number, args); };
        factories["HyperACE"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<HyperACEImpl>(in_channels, number, args); };
        factories["DownsampleConv"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<DownsampleConvImpl>(in_channels, number, args); };
        factories["FullPAD_Tunnel"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<FullPAD_TunnelImpl>(in_channels, number, args); };
        factories["DSC3k2"] = [](int in_channels, int number, std::vector<arg_complex> args) { return std::make_shared<DSC3k2Impl>(in_channels, number, args); };
    }

    auto it = factories.find(className);
    if (it != factories.end()) {
        return it->second(in_channels, number, args);
    } else {
        throw std::invalid_argument("Unknown class name: " + className);
    }
}
#endif
std::shared_ptr<DetectImpl> ModelImpl::getLastModulePtr()
{
    if (_modules.is_empty()) return nullptr;

    return (_modules[_modules->size() - 1]->name() == "DetectImpl") ?
        std::dynamic_pointer_cast<DetectImpl>(_modules[_modules->size() - 1]) : nullptr;
}

// ==================         ModelImple         =======================================

ModelImpl::ModelImpl(std::string model_file, int channels, VariantConfigs* args)
    : BaseModelImpl(model_file, channels, args)
{
    init_model();
}

void ModelImpl::load_modelsets()
{
    std::string file_ext = std::filesystem::path(cfg_file).extension().string();
    if (file_ext == ".yaml")
    {
        load_model_from_yaml();
    }
    else if (file_ext == ".pt")
    {
        auto ret = load_model_from_ptfile();
        if (std::get<0>(ret))
        {
            this->imgsz = std::get<1>(ret);
            this->scale_id = std::get<2>(ret);
            std::cout << "load imgsz: " << this->imgsz << " scale_id: " << scale_id << std::endl;
        }
    }
    else
        LOG(ERROR) << "model set file must be .yaml or .pt extension: " << file_ext ;

    parse_model_sets();
}

void ModelImpl::create_modelsets()
{
    create_modules();

    this->train();
    std::wcout << "channels: " << this->n_channels << " imgsz: " << this->imgsz << std::endl;
    torch::Tensor img_tmp = torch::zeros({ 1, this->n_channels, this->imgsz, this->imgsz});
    img_tmp = img_tmp.to(this->named_parameters().begin()->value().device());

    std::cout << "create stride , input image: " << img_tmp.sizes() << std::endl;

    torch::Tensor pred;
    std::vector<torch::Tensor> train_ret(3);
    torch::Tensor pred_seg;
    std::tie(pred, train_ret, pred_seg) = forward(img_tmp);
    std::vector<int> strides;
    for(int i = 0; i < train_ret.size(); i++) {       // [bs, na, h, w, no]
        strides.emplace_back(imgsz / train_ret[i].size(2));
    }
    auto last_module = getLastModulePtr();
    
    last_module->stride = torch::tensor(strides);
    
    stride = last_module->stride;
}

void ModelImpl::init_weights()
{
    auto last_module = getLastModulePtr();
    last_module->bias_init();
    std::cout << "bias_init: over" << std::endl;
    
    initialize_weights();
    std::cout << "initialize_weights: over" << std::endl;
}

void ModelImpl::load_weights(torch::serialize::InputArchive& in_archive)
{
    torch::serialize::InputArchive model_arch;
    if(in_archive.try_read("model", model_arch))
    {
        this->load(model_arch);
    }
    else
        LOG(WARNING) << "can't load weight from pt file.";
}

void ModelImpl::save_model(torch::serialize::OutputArchive& out_archive)
{
    c10::IValue i_imgsz(this->imgsz);
    c10::IValue i_scale_id(this->scale_id);
    c10::IValue i_task(std::string("detect"));
    out_archive.write("imgsz", i_imgsz);
    out_archive.write("scale", i_scale_id);
    out_archive.write("task", i_task);
    torch::Tensor cfg_tensor = convertMemoryToTensor(this->cfg_buffer); 
    out_archive.write("cfgs", cfg_tensor);

    torch::serialize::OutputArchive model_arch;
    this->save(model_arch);
    out_archive.write("model", model_arch);
}

torch::Tensor ModelImpl::get_stride()
{
    return this->stride;
}

int ModelImpl::get_regmax()
{
    auto last_module = getLastModulePtr();
    return last_module->reg_max;
}

void ModelImpl::load_model_from_yaml()
{
    std::string cfgfile(cfg_file);

    std::string new_yamlfile = cfg_file;
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
                if (pos != std::string::npos)
                    stem_name.erase(pos, s.length());
                filepath = fs_fullpath.parent_path().append(stem_name + fs_fullpath.extension().string()).string();
            }
        }
        return std::make_tuple(is_right, v, s, t, filepath);
        };

    auto [is_right, v, s, t, new_file] = guess_yaml_task(new_yamlfile);
    if (false == is_right)
    {
        LOG(ERROR) << "can't guess the cfg filename. ";
        exit(-1);
    }

    this->model_version = v;
    this->scale_id = s;
    this->task_name = t;

    if (b_showdebug)
    {
        std::cout << "guess cfgs: " << (is_right ? "true" : "false") << " v " << v << " sacle: " << s << " task: "
            << (t.empty() ? "Detect" : t) << std::endl;

        std::cout << "old path: " << cfgfile << "\n";
        std::cout << "exits path: " << new_file << "\n";
    }

    readModelCfgFile(new_file);
}

/*
* 从pt文件中读取训练时的重要配置，cfgs-->yaml int--> imgsz,  std::string-->scale_key
*/
std::tuple<bool, int, std::string> ModelImpl::load_model_from_ptfile()
{
    torch::serialize::InputArchive domain_arch;
    domain_arch.load_from(this->cfg_file);
    torch::Tensor read_cfgs;
    if (domain_arch.try_read("cfgs", read_cfgs))
    {
        std::cout << "read cfg tensor: " << read_cfgs.sizes() << " " << std::endl;
        convertTensorToMemory(read_cfgs, cfg_buffer);
    }
    else
    {
        LOG(ERROR) << "can't load cfgs settings from: " << cfg_file;
        return std::make_tuple(false, 0, "");
    }
    c10::IValue i_scale;
    std::string tmp_scale;  // scale_id
    if (domain_arch.try_read("scale", i_scale))
    {
        tmp_scale = i_scale.toStringRef();
    }
    else
    {
        LOG(ERROR) << "can't load scale settings from: " << cfg_file;
        return std::make_tuple(false, 0, "");
    }

    c10::IValue i_imgsz;
    int tmp_imgsz;
    if (domain_arch.try_read("imgsz", i_imgsz))
    {
        tmp_imgsz = i_imgsz.toInt();
    }
    else
    {
        LOG(ERROR) << "can't load image size settings from: " << cfg_file;
        return std::make_tuple(false, 0, "");
    }
    return std::make_tuple(true, tmp_imgsz, tmp_scale);
}

void ModelImpl::parse_model_sets()
{
    YAML::Node cfgs = YAML::Load((const char*)cfg_buffer.data());
    if(b_showdebug)
        std::cout << "cfg_buffer size: " << cfg_buffer.size() << "\n" << cfgs << std::endl;

    int nc = cfgs["nc"].as<int>();
    if (nc != n_classes) {
        LOG(WARNING) << "read nc " << nc << " no equil n_classes: " << n_classes << " check cfg yaml.";
        n_classes = nc;
    }

    if (cfgs["depth_multiple"].IsDefined())
    {   // 旧格式文件中是放在一块的
        depth_multiple = cfgs["depth_multiple"].as<float>();
        width_multiple = cfgs["width_multiple"].as<float>();
    }
    else if (cfgs["scales"].IsDefined())
    {
        if (b_showdebug)
            std::cout << "use new stype cfg yaml. \n";
        YAML::Node node_scales = cfgs["scales"];
        if (node_scales[scale_id].IsDefined())
        {
            std::vector<float> n_scale = node_scales[scale_id].as<std::vector<float>>();
            std::cout << "scale: " << scale_id << " - depth: " << n_scale[0] << " width: " << n_scale[1] << " max_channels: " << int(n_scale[2]) << std::endl;
            depth_multiple = n_scale[0];
            width_multiple = n_scale[1];
            n_maxchannels = int(n_scale[2]);
        }
    }
    else
    {
        LOG(ERROR) << "not found depth & width multiple and not found scales Node. \n";
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

    auto read_one_layer = [&](YAML::Node& cfgs)
        {
            YoloLayerDef layer;
            // froms 
            if (cfgs[0].IsSequence())
            {
                layer.froms = cfgs[0].as<std::vector<int>>();
            }
            else
                layer.froms.push_back(cfgs[0].as<int>());

            layer.number = cfgs[1].as<int>();
            layer.name = cfgs[2].as<std::string>();

            if (layer.name == "DownsampleConv" || layer.name == "FullPAD_Tunnel")
            {
                //std::cout << cfgs[3] << std::endl;
                layer.add_one_arg("None");
            }
            else
            {
                for (auto e : cfgs[3])
                {
                    if (e.IsSequence())  // 针对SPP中第二参数为数组，修正
                    {
                        std::stringstream temp;
                        temp << e.as<std::vector<float>>();
                        layer.add_one_arg(temp.str());
                    }
                    else if (e.as<std::string>() != "")
                    {
                        layer.add_one_arg(e.as<std::string>());
                    }
                    else
                    {
                        std::cout << "read yaml return empty string" << std::endl;
                        layer.add_one_arg("None");
                    }
                }
            }
            return layer;
        };

    for (int i = 0; i < node_backbone.size(); i++)
    {
        YAML::Node layer = node_backbone[i];
        layer_cfgs.push_back(read_one_layer(layer));
    }

    for (int i = 0; i < node_head.size(); i++)
    {
        YAML::Node layer = node_head[i];
        layer_cfgs.push_back(read_one_layer(layer));
    }
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor> ModelImpl::forward(torch::Tensor x)
{
    if(detect_outs.size() != 3)
    {
        detect_outs.resize(3);
    }
    if(layer_outputs.size() != _modules->size())
    {
        layer_outputs.clear();
        for(int i = 0; i < _modules->size(); i++)
            layer_outputs.push_back(torch::zeros({1}));
    }

    torch::Tensor detect_pred;
    torch::Tensor segment_pred;

    for(int i = 0; i < _modules->size()-1; i++) // no include last module -- Detect
    {
        if(b_showdebug)
            std::cout << "run layer: " << i << std::endl;

        std::vector<torch::Tensor> inputs = (i == 0) ? std::vector<torch::Tensor>{x} : get_layer_inputs(i);
        
        if(auto layer = _modules[i]->as<torch::nn::UpsampleImpl>())
        {
            layer_outputs[i] = layer->forward(inputs[0]);
        }            
        else
        {
            auto submodule = std::dynamic_pointer_cast<BaseModule>(_modules[i]);
            //std::string modulename = this->layer_cfgs[i].name;
            layer_outputs[i] = inputs.size() == 1 ? submodule->forward(inputs[0]) : submodule->forward(inputs);
        }
    
    }

    // Detect & Segment convs forward
    std::vector<torch::Tensor> last_inputs = get_layer_inputs(_modules->size()-1);
    auto last_module = getLastModulePtr();
    std::tie(detect_pred, detect_outs) = last_module->forward(last_inputs);
    segment_pred = torch::empty({0}).to(detect_pred.device());
    return { detect_pred, detect_outs, segment_pred};
}

void ModelImpl::create_modules()
{
    std::vector<int> layer_out_chs;     // 记录每层out_channels
    for (int i = 0; i < layer_cfgs.size() - 1; i++)
    {
        if(b_showdebug)
            std::cout << "\n create layer " << i << " " << layer_cfgs[i].name << std::endl;
        int ch_in = i == 0 ? this->n_channels : layer_out_chs[(i + layer_cfgs[i].froms[0])%i];  // // v13版本中的配置里，除-1, []之外，还有单个不为-1的变量
        int n = layer_cfgs[i].number;
        if(n > 1)
        {
            if(b_showdebug)
                std::cout << i << " " << layer_cfgs[i].name << " source n " << n;
            n = calculate_depth_gain(layer_cfgs[i].number, depth_multiple);
            if(b_showdebug)
                std::cout << "  new n: " << n << std::endl;
        }
        //std::cout << "init calc depth gain: " << n << std::endl;
        if(isInVector(base_modules, layer_cfgs[i].name))
        {
            int c2 = std::get<int>(layer_cfgs[i].args[0]);
            if(c2 != n_classes)
            {
                c2 = make_divisible(float(std::min(c2, n_maxchannels)) * width_multiple, 8);
            }
            layer_cfgs[i].args[0] = c2;

            if(layer_cfgs[i].name == "C3k2" ||layer_cfgs[i].name =="DSC3k2")
            {
                legacy = false;
                std::cout << " ch_in: " << ch_in << " c2: " << c2 << std::endl;
                if(scales_key[scale_id] >= scales_key["m"])
                {
                    if(b_showdebug)
                        std::cout << "C3k2: " << scale_id << " args[1] " << std::get<bool>(layer_cfgs[i].args[1]);
                    layer_cfgs[i].args[1] = true;
                    if (b_showdebug)
                        std::cout << " change to : " << std::get<bool>(layer_cfgs[i].args[1]) << "\n";
                }
            }
            if(layer_cfgs[i].name == "A2C2f")
            {
                legacy = false;

                if(scales_key[scale_id] >= scales_key["l"])
                {
                    layer_cfgs[i].args[3] = true;
                    layer_cfgs[i].args[4] = 1.5f;
                }
            }
            if(layer_cfgs[i].name == "HyperACE")
            {
                ch_in = layer_out_chs[layer_cfgs[i].froms[1]];

                int he = std::get<int>(layer_cfgs[i].args[1]);
                if(scales_key[scale_id] == scales_key["n"])
                {
                    he *= 0.5;
                }
                else if(scales_key[scale_id] == scales_key["x"])
                {
                    he *= 1.5;
                }
                layer_cfgs[i].args[1] = he;

                if(scales_key[scale_id] >= scales_key["l"])
                {
                    layer_cfgs[i].args.push_back(false);
                }
            }
        }
        else
        {
            if(layer_cfgs[i].name == "DownsampleConv")
            {
                layer_cfgs[i].args[0] = (scales_key[scale_id] >= scales_key["l"]) ? false : true;
            }
            else if(layer_cfgs[i].name == "FullPAD_Tunnel")
            {
                ch_in = layer_out_chs[(layer_cfgs[i].froms[0] + i) % i];
            }
            else if(layer_cfgs[i].name == "Concat")
            {
            }
        }

        // 根据layer_cfgs[i].name, 生成对应的Module，并压入vector中
        int total_out_ch = AppendLayer(_modules, layer_cfgs[i].name, ch_in, n, layer_cfgs[i].args);

        if(total_out_ch <= 0)   // 针对返回0的层，进行特殊处理
        {
            if(layer_cfgs[i].name == "Concat")       // 合并要将多个前面out_channel加起来算
            {
                for_each(layer_cfgs[i].froms.begin(), layer_cfgs[i].froms.end(), [&](int val){
                    total_out_ch += layer_out_chs[(val + i) % i]; 
                });
            }
            else
                LOG(ERROR) << "layer " << i << " " << layer_cfgs[i].name << " out_channel is 0." ;
        }
        if(b_showdebug)
            std::cout << "ch_in: " << ch_in << " ch_out: " << total_out_ch << std::endl;
        layer_out_chs.push_back(total_out_ch);            
    }
    // 构造 最后一层的 Detect/Segment
    YoloLayerDef final_layercfgs = layer_cfgs[layer_cfgs.size() - 1];

    std::vector<int> inchannels;
    for (auto item : final_layercfgs.froms)
        inchannels.push_back(layer_out_chs[item]);
    std::cout << "n_classes: " << n_classes << " " << inchannels << std::endl;
    if (final_layercfgs.name == "Detect")
    {
        _modules->push_back(Detect(n_classes, inchannels, legacy));
    }

    register_module("model", _modules);
}


void ModelImpl::initialize_weights()
{
    for (auto& submodule : this->modules(false))
    {
        //std::cout << submodule->name() << std::endl;
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
            submodule->as<torch::nn::BatchNorm2dImpl>()->options.eps(1e-3);
            submodule->as<torch::nn::BatchNorm2dImpl>()->options.momentum(0.03);
        }
        else if (submodule->name() == "torch::nn::LeakyReLUImpl") {
            submodule->as<torch::nn::LeakyReLU>()->options.inplace(true);
        }
        else if (submodule->name() == "torch::nn::ReLUImpl") {
            submodule->as<torch::nn::ReLU>()->options.inplace(true);
        }
        else if (submodule->name() == "torch::nn::ReLU6") {
            submodule->as<torch::nn::ReLU6>()->options.inplace(true);
        }
        else if (auto act = submodule->as<torch::nn::SiLUImpl>())
        {   // libtorch SiLU中没有对应的options成员变量
            //submodule->as<torch::nn::SiLUImpl>()->options.inplace(true);  
        }
    }
}

bool ModelImpl::readModelCfgFile(std::string yaml_file)
{
    return readFileToMemory(yaml_file, cfg_buffer);
}

bool ModelImpl::readCfgFromPtFile(std::string pt_file)
{
    torch::serialize::InputArchive domain_inputs;
    domain_inputs.load_from(pt_file);
    torch::Tensor read_cfgs;
    if (domain_inputs.try_read("cfgs", read_cfgs, true))
    {
        convertTensorToMemory(read_cfgs, cfg_buffer);
    }
    else
    {
        LOG(ERROR) << "can't load cfgs settings from: " << pt_file;
        return false;
    }
    torch::Tensor read_scale;
    if (domain_inputs.try_read("scale", read_scale, true))
    {
        
    }
    else
    {
        LOG(ERROR) << "can't load scale settings from: " << pt_file;
        return false;
    }
    return true;
}

