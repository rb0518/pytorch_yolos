#include "yaml_load.h"

#include <charconv>
#include <ostream>
#include <tuple>
#include <vector>
#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <regex>
#include "utils.h"

void TestYamlRead(const std::string& yamlfile, bool debug_show)
{
    YAML::Node cfgs = YAML::LoadFile(yamlfile);
    //std::cout << cfgs << std::endl;
    
    int nc = cfgs["nc"].as<int>();
    float depth_multiple = cfgs["depth_multiple"].as<float>();
    float width_multiple = cfgs["width_multiple"].as<float>();

    // load anchors
    std::vector<std::vector<int>> anchors;
    YAML::Node node_anchors = cfgs["anchors"];   
    
    std::vector<YoloLayerDef> layer_cfgs;

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
        for(auto e : cfgs[3])
            layer.add_one_arg(e.as<std::string>());

        return layer;
    };

    if(!node_anchors.IsNull() && node_anchors.IsSequence())
    {
        for(YAML::const_iterator it = node_anchors.begin(); it != node_anchors.end(); ++it){
            if((*it).IsSequence()){
                anchors.push_back(it->as<std::vector<int>>());
            }
        }
    }
    YAML::Node node_backbone = cfgs["backbone"];
    YAML::Node node_head = cfgs["head"];

    int total_layers = node_backbone.size() + node_head.size();

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

    if(debug_show)
    {
        std::cout << "nc: " << nc << std::endl;

        std::cout <<"Total layers: " << layer_cfgs.size() << std::endl;
        std::for_each(layer_cfgs.begin(),layer_cfgs.end(),[](YoloLayerDef layer){
            std::cout << layer.show_info() << std::endl;
            });  
    }
}

bool TestDataYamlRead(const std::string& yamlfile, bool debug_show/* = false*/)
{
    std::cout << "TestDataYamlRead: " << yamlfile << std::endl;
    if(std::filesystem::exists(yamlfile))
    {
        LOG(ERROR) << "Select " << yamlfile << " not exists.";
    }

    YAML::Node cfgs = YAML::LoadFile(yamlfile);

    std::string data_root = cfgs["path"].as<std::string>();
    std::string train_dir = cfgs["train"].as<std::string>();
    std::string val_dir = cfgs["val"].as<std::string>();

    std::string test_dir = "";
    if(cfgs["test"].IsNull())
        test_dir = cfgs["test"].as<std::string>();
    
    int nc = cfgs["nc"].as<int>();

    std::vector<std::string> names;
    names = cfgs["names"].as<std::vector<std::string>>();


    std::cout << "path:" << data_root << std::endl;
    std::cout << "train_dir: " << train_dir << std::endl;
    std::cout << "val_dir" << val_dir << std::endl;

    std::cout << "names: " << names.size() << std::endl;
    if(debug_show)
    {
        for(auto item : names)
        {
            std::cout << item << std::endl;
        }
    }

    std::vector<std::string> train_imagefiles;
    std::vector<std::string> train_labelfiles;
    std::string check_folder = std::filesystem::path(data_root).append(train_dir).string();

    std::string train_lable_folder;
    std::string train_image_folder;
    bool b_loadfromdir = false;

    if(std::filesystem::is_directory(std::filesystem::path(check_folder)))  // 指定的是目录
    {
        b_loadfromdir = true;
        std::string cur_data_name = std::filesystem::path(check_folder).filename().string(); // train2017 
        std::cout << "will load earch image file from: " << check_folder << "data type: " << cur_data_name << std::endl;
        train_image_folder = check_folder;
        train_lable_folder = std::filesystem::path(check_folder).parent_path().parent_path().append("labels").append(cur_data_name).string();
        std::cout << "Label file in dir: " << train_lable_folder << std::endl;
    }

    if(b_loadfromdir)
    {
        searchfiles_in_folder(train_image_folder, ".jpg", train_imagefiles);
        searchfiles_in_folder(train_lable_folder, ".txt", train_labelfiles);
        if(train_imagefiles.size() != train_labelfiles.size())
        {   // 简单检测，最终能用的版本应该考虑到文件不一致，要从列表中删除掉不存在的文件
            std::cout << "find imag files: " << train_imagefiles.size() << " and label files: " << train_labelfiles.size() << std::endl;
            for(int i = 0; i < std::min(train_imagefiles.size(), train_labelfiles.size()); i++)
                std::cout << train_imagefiles[i] << " -- " << train_labelfiles[i] << std::endl;
        }
    }   
    else 
    {   // 是.txt文件列表，直接读取，暂时未写

    }


    return true;
}


VariantConfigs set_cfg_opt_default()
{
    VariantConfigs opts;
    opts["weights"] = "";
    opts["weights"]= "";
    opts["cfg"]= "models/yolov5s.yaml";
    opts["data"]= "data/coco128.yaml";
    opts["hyp"]= "data/hyp.scratch.yaml";
    opts["epochs"]= 300;
    opts["batch_size"]= 16;
    opts["img_size"]= std::vector<int>({ 640, 640 });
    opts["rect"]= false;
    opts["resume"]= false;
    opts["nosave"]= false;
    opts["notest"]= false;
    opts["noautoanchor"]= false;
    opts["evolve"]= false;
    opts["bucket"]= "";
    opts["cache_images"]= false;
    opts["image_weights"]= false;
    opts["device"]= "";
    opts["multi_scale"]= false;
    opts["single_cls"]= false;
    opts["adam"]= false;
    opts["sync_bn"]= false;
    opts["local_rank"]= -1;
    opts["workers"]= 8;
    opts["project"]= "runs/train";
    //entity: null
    opts["name"]= "exp";
    opts["exist_ok"]= false;
    opts["quad"]= false;
    opts["linear_lr"]= false;
    opts["label_smoothing"]= 0.0f;
    opts["upload_dataset"]= false;
    opts["bbox_interval"]= -1;
    opts["save_period"]= -1;
    opts["artifact_alias"] = "latest";
    opts["world_size"]= 1;
    opts["global_rank"]= -1;
    opts["save_dir"]= "runs/train/exp";
    opts["total_batch_size"]= 16;

    return opts;
}

VariantConfigs load_cfg_yaml(const std::string& cfgs_file)
{
    VariantConfigs cfgs;
//    set_cfg_opt_default(opts);
    /*
    std::ifstream fin(cfgs_file);
    if (!fin) {
        LOG(ERROR) << "Failed to open file: " << cfgs_file;
    }

    std::stringstream str_stream;
    str_stream << fin.rdbuf();*/
    YAML::Node doc = YAML::LoadFile(cfgs_file);
    for (auto it = doc.begin(); it != doc.end(); ++it)
    {
        const std::string& key = it->first.as<std::string>();
        const YAML::Node& value_node = it->second;


        if (cfgs.count(key) > 0)
        {
            //using Value = std::variant<std::string, int, bool, float, std::vector<int>>;
            auto v = cfgs[key];
            auto type_index = v.index();
            switch (type_index)
            {
            case 0:
                v =  value_node.as<std::string>();
                break;
            case 1:
                v = value_node.as<int>();
                break;
            case 2:
                v = value_node.as<bool>();
                break;
            case 3:
                v = value_node.as<float>();
                break;
            case 4:
                v = value_node.as<std::vector<int>>();
                break;
            }
            cfgs[key] = v;
        }
        else
        {
            if (value_node.IsScalar()) {
                auto check_node_scalartype = [&](YAML::Node node) {
                    std::string str_value = node.as<std::string>();
                    //std::cout << key << " : " << str_value << std::endl;
                    std::regex int_regex("^-?\\d+$");
                    std::regex float_regex("^-?\\d+(\\.\\d+)?([eE][-+]?\\d+)?$");
                    if (std::regex_match(str_value, int_regex)) 
                    {
                    //    std::cout << "int" << std::endl;
                        return 1;
                    }
                    else if (std::regex_match(str_value, float_regex)) {
                    //    std::cout << "float" << std::endl;
                        return 2;
                    }
                    return 0;
                    };
                auto scalartype = check_node_scalartype(value_node);
                switch (scalartype)
                {
                case 1:
                    cfgs[key] = value_node.as<int>(); 
                    break;
                case 2:
                    cfgs[key] = value_node.as<float>(); 
                    break;
                default:
                    std::string tmpstr = value_node.as<std::string>();
                    if (tmpstr == "True" || tmpstr == "true")
                        cfgs[key] = true;
                    else if(tmpstr == "False" || tmpstr == "false")
                        cfgs[key] = false;
                    else
                        cfgs[key] = tmpstr; 
                    break;
                }
            }
            else if (value_node.IsSequence())
            {
                cfgs[key] = value_node.as<std::vector<int>>();
            }
        }
    }
    return cfgs;
}
#include <sstream>
//#include <iomanip> 通过指定保留位能避免科学表达，但要判定截断出错，暂不考虑
void save_cfg_yaml(const VariantConfigs& cfgs, const std::string& optfile)
{
    YAML::Node node;
    for (const auto& [key, value] : cfgs) {
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::vector<int>>) {
                YAML::Node seq;
                for (const auto& item : arg) seq.push_back(item);
                node[key] = seq;
            }
            else if constexpr (std::is_same_v<T, float>){
                    // 解决存储时，1.0 --> 1的问题
                    std::stringstream ss;
                    ss << arg;
                    //std::cout << key << " " << ss.str() ;
                    std::string s = ss.str();
                    #if 0                    
                    std::regex int_regex("^-?\\d+$");   // 只判定整数，简化科学计算之类的要截断误差
                    if (std::regex_match(s, int_regex))     
                    #else
                    if(s.find('.') ==  std::string::npos)
                    #endif
                        s = s + ".0";       
                    //std::cout << " : " << s << std::endl;                                 
                    node[key] = s;
            }
            else 
            {
                 node[key] = arg;
            }
            }, value);
    }

    std::ofstream fout(optfile);
    fout << node;
}

VariantConfigs set_cfg_hyp_default()
{
    VariantConfigs hyps;

    hyps["lr0"] = 0.01f;   // initial learning rate(SGD = 1E-2, Adam = 1E-3)
    hyps["lrf"] = 0.2f;  // final OneCycleLR learning rate(lr0 * lrf)
    hyps["momentum"] = 0.937f;  // SGD momentum / Adam beta1
    hyps["weight_decay"] = 0.0005f;  // optimizer weight decay 5e-4
    hyps["warmup_epochs"] = 3.0f;  // warmup epochs(fractions ok)
    hyps["warmup_momentum"] = 0.8f;  // warmup initial momentum
    hyps["warmup_bias_lr"] = 0.1f;  // warmup initial bias lr
    hyps["box"] = 0.05f;  // box loss gain
    hyps["cls"] = 0.5f;  // cls loss gain
    hyps["cls_pw"] = 1.0f;  // cls BCELoss positive_weight
    hyps["obj"] = 1.0f;  // obj loss gain(scale with pixels)
    hyps["obj_pw"] = 1.0f;  // obj BCELoss positive_weight
    hyps["iou_t"] = 0.20f;  // IoU training threshold
    hyps["anchor_t"] = 4.0f;  // anchor - multiple threshold
    //hyps["anchors"]= 3;  // anchors per output layer(0 to ignore)
    hyps["fl_gamma"] = 0.0f;  // focal loss gamma(efficientDet default gamma = 1.5)
    hyps["hsv_h"] = 0.015f;  // image HSV - Hue augmentation(fraction)
    hyps["hsv_s"] = 0.7f;  // image HSV - Saturation augmentation(fraction)
    hyps["hsv_v"] = 0.4f;  // image HSV - Value augmentation(fraction)
    hyps["degrees"] = 0.0f;  // image rotation(+/ -deg)
    hyps["translate"] = 0.1f;  // image translation(+/ -fraction)
    hyps["scale"] = 0.5f;  // image scale(+/ -gain)
    hyps["shear"] = 0.0f;  // image shear(+/ -deg)
    hyps["perspective"] = 0.0f;  // image perspective(+/ -fraction), range 0 - 0.001
    hyps["flipud"] = 0.0f;  // image flip up - down(probability)
    hyps["fliplr"] = 0.5f;  // image flip left - right(probability)
    hyps["mosaic"] = 1.0f;  // image mosaic(probability)
    hyps["mixup"] = 0.0f;  // image mixup(probability)

    return hyps;
}


void show_cfg_info(const std::string& title, const VariantConfigs& cfgs)
{
    std::cout << "\x1b[31m" << title << ": " << "\x1b[0m";
    for (const auto& [key, value] : cfgs) 
    {
        std::cout << key << ": ";
        int type_index = value.index();
        if (type_index == 0) {
            std::cout << std::get<std::string>(value) << " ";
        }
        else if (type_index == 1) {
            std::cout << std::get<int>(value) << " ";
        }
        else if (type_index == 2) {
            if (std::get<bool>(value) == true)
                std::cout << "true ";
            else
                std::cout << "false ";
        }
        else if (type_index == 3) {
            std::stringstream ss;
            ss << std::get<float>(value);
            std::string s = ss.str();
            if (s.find('.') == std::string::npos)
                s = s + ".0";
            std::cout << s << " ";
        }
        else if (type_index == 4) {
            std::vector<int> tmp = std::get<std::vector<int>>(value);
            std::cout << "[ ";
            for (int i = 0; i < tmp.size(); i++)
                std::cout << tmp[i] << " ";
            std::cout << "] ";
        }
    }
    std::cout << std::endl;
}

void read_data_yaml(const std::string& data_file, std::string& train_path, std::string& val_path, std::vector<std::string>& names)
{
    YAML::Node data_dict = YAML::LoadFile(data_file);
    YAML::Node dict_name = data_dict["names"];

    bool new_type = dict_name.IsMap();
    if(new_type)
    {
        for (YAML::const_iterator it = dict_name.begin();
            it != dict_name.end(); ++it)
        {
            names.emplace_back(it->second.as<std::string>());
        }
    }
    else
    {
        names = dict_name.as<std::vector<std::string>>();
    }

    if(!data_dict["path"].IsDefined())
    {
        train_path = data_dict["train"].as<std::string>();
        val_path = data_dict["val"].as<std::string>();
    }
    else
    {
        std::string data_path = data_dict["path"].as<std::string>();
        train_path = std::filesystem::path(data_path).append(data_dict["train"].as<std::string>()).string();
        val_path = std::filesystem::path(data_path).append(data_dict["val"].as<std::string>()).string();
    }


    auto delete_end_backslash = [](std::string& s) {
            auto s_tmp = s;
            if(s.substr(s.length()-1)=="\\" || s.substr(s.length()-1)=="/")
            {
                s_tmp = s.substr(0, s.length()-1);
            }
            return s_tmp;
        };    
    train_path = delete_end_backslash(train_path);  
    val_path = delete_end_backslash(val_path);
}

