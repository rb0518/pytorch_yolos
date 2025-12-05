#include <filesystem>

#include "BaseModel.h"
#include "utils.h"
#include "torch_utils.h"
// ===================== Base Model =====================================
BaseModelImpl::BaseModelImpl(std::string model_file, int channels, VariantConfigs* args)
{
    cfg_file = model_file;
    n_channels = channels;
    set_args_prt(args);
}

void BaseModelImpl::init_model()
{
    this->load_modelsets();
    this->create_modelsets();
    this->init_weights();
}

void BaseModelImpl::set_args_prt(VariantConfigs* _args)
{
    if (_args == nullptr)
        return;

    p_args = _args;
    imgsz = p_args->count("imgsz") > 0 ? std::get<int>(p_args->at("imgsz")) : 640;

    //b_showdebug = p_args->count("debug_show_info") > 0 ? std::get<bool>(p_args->at("debug_show_info")) : false;
}

void BaseModelImpl::set_classes(std::vector<std::string>& cls_names)
{
    if (cls_names.size() == 0)
    {
        this->n_classes = 1;
        this->names.push_back("item");
    }

    for (auto& item : cls_names)
    {
        names.push_back(item);
    }
    this->n_classes = names.size();
}

std::vector<torch::Tensor> BaseModelImpl::get_layer_inputs(int layer_id)
{
    std::vector<torch::Tensor> inputs;
    for_each(this->layer_cfgs[layer_id].froms.begin(), this->layer_cfgs[layer_id].froms.end(), [&](int from){
        inputs.push_back(this->layer_outputs[(layer_id + from)%layer_id]);});
    return inputs;
}

void BaseModelImpl::show_modelinfo(bool show_net)
{
    std::cout << ColorString("Model: ", "Info") << this->cfg_file << std::endl;
    //std::cout << "nc: " << n_classes << " ch: " << n_channels << " height: " << image_height << " widht: " << image_width << std::endl;
    int i = 0;
    std::for_each(layer_cfgs.begin(), layer_cfgs.end(), [&](YoloLayerDef layer) {
        std::cout << std::setw(3) << i << "  " << layer.show_info() << std::endl; i += 1;
        });

    std::cout << "Total layers: " << layer_cfgs.size() << std::endl;

    if (show_net)
    {
        int layer_id = 0;
        for (auto m : named_modules(std::string(), false))
        {
            std::cout << "layer_id " << layer_id << std::endl;
            module_sub_info(*(m.value()), m.key(), 0);
            layer_id += 1;
        }
    }
}
