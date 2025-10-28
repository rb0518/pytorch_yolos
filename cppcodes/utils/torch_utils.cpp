#include "utils.h"
#include "torch_utils.h"

torch::Device get_device(const std::string& dev_set)
{
    std::string device_define = dev_set;
    torch::Device device = torch::Device(torch::kCPU);

    int device_count = torch::cuda::device_count();

    // "CPU" or "CUDA" or "0" ..."2"
    std::transform(device_define.begin(), device_define.end(), device_define.begin(),
        [](unsigned char c) {return std::toupper(c); });

    auto [idx, is_ok] = ConvertToNumber(device_define);
    if (is_ok)
    {
        int gpu_idx = int(idx);
        if (gpu_idx >= 0 && gpu_idx < device_count)
            device = torch::Device(c10::DeviceType::CUDA, gpu_idx);
        else if (device_count)
        {
            LOG(WARNING) << "select GPU id: " << gpu_idx << " wrong. use default device 0.";
            device = torch::Device(c10::DeviceType::CUDA);
        }
        else
        {
            device = torch::Device(torch::kCPU);
        }
    }
    else if (device_define == "CPU" || device_define == "CUDA" || device_define == "GPU")
    {
        if ((device_define == "GPU" || device_define == "CUDA") && device_count)
        {
            device = torch::Device(c10::DeviceType::CUDA);
        }
        else
        {
            std::cout << ColorString("Device: ") << "CPU \n";
            device = torch::Device(torch::kCPU);
        }
    }
    else
    {
        if (device_count)
        {
            LOG(WARNING) << "your select device type: " << dev_set << " wrong. use default device 0.";
            device = torch::Device(c10::DeviceType::CUDA);
        }
        else
        {
            LOG(WARNING) << "your select device type: " << dev_set << " wrong. use default CPU.";
            device = torch::Device(c10::DeviceType::CPU);
        }
    }


    std::cout << "get_device return device type: " << device.type() << std::endl;
    return device;
}


std::tuple<std::shared_ptr<torch::optim::Optimizer>, int, float> smart_optimizer(std::shared_ptr<ModelImpl> ptr_model, VariantConfigs& args)
{
    auto model = ptr_model.get();
    const int nbs = 64;
    int batch_size = std::get<int>(args["batch"]);
    int accumulate = std::max(static_cast<int>(std::round(nbs / batch_size)), 1);
    float lr0 = std::get<float>(args["lr0"]);
    float momentum = std::get<float>(args["momentum"]);
    float weight_decay = std::get<float>(args["weight_decay"]);
    weight_decay = weight_decay * float(batch_size * accumulate / nbs);

    std::shared_ptr<torch::optim::Optimizer> optimizer{ nullptr };
    std::vector<torch::Tensor> pg0, pg1, pg2;
    //std::cout << "model named_modules size: " << model->named_modules("", false).size() << std::endl;
    for (const auto& module : model->named_children()) 
    {
        for (auto pair : module.value()->named_parameters())
        {
            //std::cout << pair.key() <<": " << module.key() << " " << module.value()->name() << std::endl;
            if (pair.key().find(".bias") != std::string::npos) // no decay
                pg2.push_back(pair.value());
            else if (pair.key().find(".weight") != std::string::npos) {
                //if (auto* bn_module = module.value()->as<torch::nn::BatchNorm2d>())
                if (pair.key().find("bn.weight") != std::string::npos)
                    pg1.push_back(pair.value());  // no decay
                else
                    pg0.push_back(pair.value());  // with decay
            }
        }
    }

    if (std::get<std::string>(args["optimizer"]) == "adam")
    {
         
        std::cout << "Adam" << " lr0 " << lr0 << std::endl;
        auto adam_option = torch::optim::AdamOptions(lr0).betas(std::make_tuple(momentum, 0.999)).weight_decay(0.0);
        optimizer = std::make_shared<torch::optim::Adam>(
            pg2, adam_option);

        if (!pg0.empty()) {
            std::unique_ptr<torch::optim::AdamOptions> pg0_options = std::make_unique<torch::optim::AdamOptions>(lr0);
            pg0_options->weight_decay(weight_decay);
            torch::optim::OptimizerParamGroup paramgroup_pg0(pg0);
            paramgroup_pg0.set_options(std::move(pg0_options));

            optimizer->add_param_group(paramgroup_pg0);
        }

        if (!pg1.empty()) {
            std::unique_ptr<torch::optim::AdamOptions> pg1_options = std::make_unique<torch::optim::AdamOptions>(lr0);
            pg1_options->weight_decay(0.0);
            torch::optim::OptimizerParamGroup paramgroup_pg1(pg1);
            paramgroup_pg1.set_options(std::move(pg1_options));            
            optimizer->add_param_group(paramgroup_pg1);
        }
    }
    else
    {   // default at here -- opt.adam : false
        std::cout << "SGD" << " lr0 " << lr0 << std::endl;
        optimizer = std::make_shared<torch::optim::SGD>(
            pg2, torch::optim::SGDOptions(lr0)
            .momentum(momentum)
            .nesterov(true).weight_decay(0.0));

        if (!pg0.empty()) {
            std::unique_ptr<torch::optim::SGDOptions> pg0_options = std::make_unique<torch::optim::SGDOptions>(lr0);
            pg0_options->weight_decay(weight_decay);
            torch::optim::OptimizerParamGroup paramgroup_pg0(pg0);
            paramgroup_pg0.set_options(std::move(pg0_options));

            optimizer->add_param_group(paramgroup_pg0);
        }
        if (!pg1.empty()) {
            std::unique_ptr<torch::optim::SGDOptions> pg1_options = std::make_unique<torch::optim::SGDOptions>(lr0);
            pg1_options->weight_decay(0.0);
            torch::optim::OptimizerParamGroup paramgroup_pg1(pg1);
            paramgroup_pg1.set_options(std::move(pg1_options));

            optimizer->add_param_group(paramgroup_pg1);

        }
    }
    // 按python代码，显示optimizer groups信息
    std::cout << "optimizer with  groups pg1: " << pg1.size() << " weight(decay=0.0), pg0: "
        << pg0.size() << " weight(decay= " << weight_decay << ") pg2 (bias): "
        << pg2.size() << std::endl;
    
    return {optimizer, accumulate, lr0};
}


void save_checkpoint(std::string filename, torch::nn::Module* model,
    std::shared_ptr<torch::optim::Optimizer> optimizer, int epoch)
{
    torch::serialize::OutputArchive ckpt_out;

    torch::serialize::OutputArchive model_out;
    model->save(model_out);
    ckpt_out.write("model", model_out);
    torch::serialize::OutputArchive optim_out;
#if 0
    optimizer->save(optim_out);
#else   // 保持与pytorch中pt一致
    int pg_size = optimizer->param_groups().size();
    optim_out.write("size", torch::tensor(pg_size));
    for (int i_pg = 0; i_pg < optimizer->param_groups().size(); i_pg++)
    {
        torch::serialize::OutputArchive tmp_pg;
        optimizer->param_groups()[i_pg].options().serialize(tmp_pg);
        optim_out.write("param_groups/" + std::to_string(i_pg), tmp_pg);
    }
#endif
    ckpt_out.write("optim", optim_out);

    ckpt_out.write("epoch", torch::tensor(epoch));
    ckpt_out.save_to(filename);
}

ModelEMA::ModelEMA(std::shared_ptr<Model> ptr_model, float decay, int updates)
{
    decay_ = decay;
    updates_ = updates;
    auto model = ptr_model->get();
    // const std::string& yaml_file, int classes, int imagewidth, int imageheight, int channels, bool showdebuginfo
    ema_model_ = std::make_shared<Model>(model->cfgfile, model->n_classes, 
        model->image_width, model->image_height, model->n_channels, model->b_showdebug);
    (*ema_model_)->eval();

    for (auto& param : (*ema_model_)->parameters()) {
        param.set_requires_grad(false);
    }
}

void ModelEMA::update(torch::nn::Module& model)
{
    (*ema_model_)->to(model.parameters().begin()->device());

    {
        torch::NoGradGuard nograd;
        updates_ += 1;
        float d = decay_function(updates_);

        std::unordered_map<std::string, torch::Tensor> model_params;
        for (const auto& param : model.named_parameters())
        {
            model_params[param.key()] = param.value();
            // std::cout << "model_params key: " << param.key() << std::endl;
        }

        for (auto& param : (*ema_model_)->named_parameters())
        {
            std::string str_name = param.key();
            // std::cout << "ema_model_ key: " << str_name << std::endl;

            if (param.value().is_floating_point())
            {
                auto v = param.value().clone();
                v = v * d;
                v = v + (1.f - d) * model_params[str_name].detach();
                param.value().data().copy_(v.data());
            }
        }
    }
}
