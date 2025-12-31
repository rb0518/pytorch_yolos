#include <filesystem>
#include <tuple>
#include <fstream>
#include <cmath>

#include "train.h"
#include "utils.h"

#include "yolo.h"
#include "general.h"

#include "loss.h"
#include "lambda_lr.h"

#include "progressbar.h"
#include "plots.h"
#include "yaml_load.h"
#include "torch_utils.h"
#include "nms.h"
//#include "augmentations.h"

std::function<float(int)> create_lr_lambda(bool linear_lr, float lrf, int epochs) {
    if (linear_lr) {
        // 线性插值
        return [lrf, epochs](int x) -> float {
            return (1.0f - static_cast<float>(x) / (epochs)) * (1.0f - lrf) + lrf;
            };
    }
    else {
        // 余弦插值
        return [lrf, epochs](int x) -> float {
            return ((1 - std::cos(x * M_PI / epochs)) / 2) * (lrf - 1) + 1;
            };
    }
}

// ------------------------  start BaseTrain ------------------
BaseTrainer::BaseTrainer( VariantConfigs cfg_default, std::string root_dir)
    : m_rootDir(root_dir), m_args(cfg_default)
{
    getArgsSettings();
    checkResumeSettings();
    show_cfg_info("BaseTrainer cfg: ", m_args);

    initDevice();
    init_torch_seek(std::get<int>(m_args["seed"]) + 1);

    initDirs();

    save_cfg_yaml(m_args, std::filesystem::path(m_saveDir).append("args.yaml").string());
}

void BaseTrainer::setupTrainer()
{
    getSpecialSettins();    

    setupModel();
    freezeModules();

    setupDataloader();

    setupOptimizer();
    setupScheduler();

    loadPretrained();
    resumeLastTraining();

    createComputeLoss();

    createValidator();
}

void BaseTrainer::doTrainTask()
{
    int nb = train_loader->getNumberOfBatch(); // number of batchs
    int nw = -1;
    int nbs = std::get<int>(m_args["nbs"]);
    int warmup_epochs = std::get<float>(m_args["warmup_epochs"]);
    if (warmup_epochs > 0)
        nw = std::max(warmup_epochs * nb, 100); // 

    int base_epochs = epochs - std::get<int>(m_args["close_mosaic"]);
    int ni = 0;
    // check parameters and optimizer params
    progressbar pbar(nb, 30);
    pbar.set_done_char(std::string("█"));
    std::cout << ColorString("Start training:") << ", start_epoch: " << start_epochs <<" end epochs: " << epochs << " number of batch: " << nb << std::endl;
    optimizer->zero_grad();
    for (int epoch = start_epochs; epoch < epochs; epoch++)
    {
        train_loader->closeMosaic(epoch >= base_epochs);       
        //scheduler->step();
        model->train();
        m_compute_loss->reset();

        std::cout << m_compute_loss->getLossitemsNames() << std::endl;
        int idx_epoch = 0;
        pbar.reset();
        for(auto itr = train_loader->getBegin(); itr != train_loader->getEnd(); ++itr)
        {
            auto batch = *itr;
            save_batch_sample_image(epoch, base_epochs, start_epochs, batch);
            ni = idx_epoch + nb * epoch;

            if (ni <= nw)
                doWarmup(epoch, ni, nw, nbs, train_loader->getBatchSize());
                    
            preprocess_batch(batch);

            // 经测试，须按zero_grad, backward, step顺序来，如果采取pytorch一样，后面2、4次累计造成不收敛，
            // 可能需要保留一些额外信息，暂时没有查找到相关资料
            auto [pred_t, train_v, mask_t] = model->forward(std::get<torch::Tensor>(batch[GET_VDK(VDK::IMG)]));
            auto [loss, loss_items] = m_compute_loss->forward(train_v, batch);
            m_compute_loss->addNewItems(loss, loss_items);
            optimizer->zero_grad();
            loss.backward();

            // 经测试，无法像pytorch代码中一样，采用累计方式，所以只能按标准的zero_grad, backward, step进行
            optimizer->step();

            auto num_targets = std::get<torch::Tensor>(batch[GET_VDK(VDK::BBOXES)]).size(0);
            pbar.set_prefix(refresh_batch_end_info(epoch, epochs, idx_epoch, num_targets));
            idx_epoch+=1;
            pbar.update();
        }
        scheduler->step(); 
        std::cout << std::endl << std::endl;

        on_train_epoch_over(epoch);

        if (m_bVal || m_validator != nullptr)    // # (bool) validate/test during training
        {   // add call valitate class do_valitate at here
            m_validator->doValidateTask();
        }
    }
    on_train_over();
}

void BaseTrainer::getArgsSettings()
{
    m_bPlotsEnable = getVariantValue<bool>(&m_args, "plots", true);
    m_modelFilename = std::filesystem::path(m_rootDir).append(std::get<std::string>(m_args["model"])).string();
    m_bResume = getVariantValue<bool>(&m_args, "resume", false);

    m_bVal = getVariantValue<bool>(&m_args, "val", false);
}

/*
* 如果是启动resume训练，读取原来训练保存目录下的cfg配置文件
* 由于C++代码没法让像python一样使用动态变量，所以必须由weight变量来
* 传递具体的路径
*/
void BaseTrainer::checkResumeSettings()
{
    m_bResume = (m_bResume && std::filesystem::path(m_modelFilename).extension() == ".pt") ? true : false;

    if (m_bResume)  // 两样检测正确
    {
        std::string weight_file = std::get<std::string>(m_args["weights"]);
        auto wf_parent = std::filesystem::path(m_modelFilename).parent_path();
        auto last_opts_file = wf_parent.append("args.yaml");
        if (std::filesystem::exists(last_opts_file)) {
            // 读取上次配置，并替换当前args中相关设置
            auto last_opts = load_cfg_yaml(last_opts_file.string());

            m_args["data"] = std::get<std::string>(last_opts["data"]);
            m_args["imgsz"] = std::get<int>(last_opts["imgsz"]);
            m_args["batch"] = std::get<int>(last_opts["batch"]);
            m_args["device"] = std::get<std::string>(last_opts["device"]);
            m_args["close_mosaic"] = std::get<int>(last_opts["close_mosaic"]);
            // 简化不做比对，直接使用上次的设置的model cfg配置
            m_args["model"] = std::get<std::string>(last_opts["model"]);
        }
        else
        {
            LOG(ERROR) << "resume train, but not found the last time args.yaml file: " << last_opts_file.string();
            m_args["resume"] = false;
        }
    }
    else
    {
        LOG(ERROR) << "want resume train, must args.resume and model_file must be .pt file.";
    }
}

void BaseTrainer::initDevice()
{
    std::cout << "input device type: " << std::get<std::string>(m_args["device"]) << "\n";
    m_device = get_device(std::get<std::string>(m_args["device"]));
    std::cout << ColorString("Device: ") << m_device.type() << std::endl;
}

void BaseTrainer::initDirs() 
{
    std::tie(m_saveDir, m_lastPtFile, m_bestPtFile, m_resultsFile) = getSaveDirAndFilenames(m_args, m_rootDir);
    std::cout << ColorString("save dir: ", "info") << m_saveDir << "\n";
}

void BaseTrainer::loadPretrained()
{
//    resume_load_ok = false;
    bool load_from_jit = false;
    std::string jit_weights_file = std::get<std::string>(m_args["jit_weights"]);
    if (jit_weights_file != "")
    {
        auto path = std::filesystem::path(m_rootDir).append(jit_weights_file);
        if (std::filesystem::exists(path))
        {
            load_from_jit = true;
            jit_weights_file = path.string();

            LoadWeightFromJitScript(jit_weights_file, *model, true);
        }
    }
    if (m_bResume)
    {
         std::cout << ColorString("Load pretrain weights: ", "Info") << m_modelFilename;
        
        torch::serialize::InputArchive ckpt_in;
        ckpt_in.load_from(m_modelFilename);
        model->load_weights(ckpt_in);
        std::cout << " over." << std::endl;
    }
}

void BaseTrainer::freezeModules()
{
    std::vector<std::string> freeze;
    freeze.push_back(".dfl");
    for (auto p : model->named_parameters())
    {
        auto ret_str = findSubstringInStrings(p.key(), freeze);
        if (ret_str != "")
        {
            p.value().requires_grad_(false);
            std::cout << "freezeing: " << p.key() << std::endl;
        }
        else if(!p.value().requires_grad() && torch::is_floating_point(p.value()))
        {
            p.value().requires_grad_(true);
        }
    }
}

void BaseTrainer::setupOptimizer()
{
    int nbs = std::get<int>(m_args["nbs"]);
    int batch_size = train_loader->getBatchSize();
    accumulate = std::max(int(std::roundf(float(nbs) / float(batch_size))), 1);

    float weight_decay = std::get<float>(m_args["weight_decay"]);
    std::cout << "read weigth_decay : " << weight_decay << " accumulate " << accumulate << " batch_size: " << batch_size  << std::endl;
    weight_decay = weight_decay * float(batch_size) * float(accumulate) / float(nbs);
    int iterations = std::ceil(train_loader->getTotalSamples() / std::max(batch_size, nbs))* std::get<int>(m_args["epochs"]);

    std::string name = getVariantValue<std::string>(&m_args, "optimizer", "SGD");
    float lr = getVariantValue<float>(&m_args, "lr0", 0.1);
    float momentum = getVariantValue<float>(&m_args, "momentum", 0.1);
    std::cout << "new weigth_decay: " << weight_decay << " iterations: " << iterations << " name: " << name << std::endl;

    this->optimizer = build_optiomizer(model->n_classes, name, lr, momentum, weight_decay, iterations);
    if(name == "auto")
       m_args["warmup_bias_lr"] = 0.0f;
}

std::shared_ptr<torch::optim::Optimizer> BaseTrainer::build_optiomizer(int nc,std::string optim_name, 
    float _lr, float _momentum, float _decay, int iterations)
{
    std::string name = optim_name;
    double lr = _lr;
    double momentum = _momentum;
    double decay = _decay;
    if(name == "auto")
    { 
        std::cout << ColorString("optimizer:", "info") << "optimizer = auto found, ";
        std::cout << "ignoring lr0 = " << lr <<
            " and momentum = " << momentum << " and "
            << "determining best \'optiomizer\', \'lr0\' and \'momentum\' automatically..." << std::endl;

        auto round_to_decimal = [](double num_d, int f){
            double factor = std::pow(10.0, f);
            return std::round(num_d * factor) / factor;
        };

        double lr_fit =round_to_decimal(double(0.002 * 5.0 / double(4 + nc)), 6);

        if (iterations > 10000)
        {
            name = "SGD";
            lr = 0.01f;
            momentum = 0.9f;
        }
        else{
            name = "AdamW";
            lr = lr_fit;
            momentum = 0.9f;
        }
    }

    std::vector<torch::Tensor> pg0, pg1, pg2;
    for (auto& module : model->named_modules(std::string(), false))
    {
        for(auto& pair : module.value()->named_parameters(false))
        {
            if (pair.key().find("bias") != std::string::npos) // no decay
            {
                pg2.push_back(pair.value());
            }
            else if (auto m = module.value()->as<torch::nn::BatchNorm2d>())
            {
                pg1.push_back(pair.value());  // no decay
            }
            else
            {
                pg0.push_back(pair.value());  // with decay
            }
        }
    }

    if (name == "Adamax" || name == "Adamax" || name == "NAdam" || name == "NAdam")// 其它情况:Adam, Adamax, NAdam NAdam在optim中没有，均用Adam
        name = "Adam";

    optimizer_name = name;
    std::cout << ColorString("optimizer: ", "info") << name << " (lr = " << lr
        << " momentum: " << momentum << ") with parameter groups g[1]: "
        << pg1.size() << " weight(decay=0.0) g[0]: " << pg0.size() << " weight(decay = "
        << decay << " ) g[2]: " << pg2.size() << " bias(decay=0.0)" << std::endl;

    std::vector<torch::optim::OptimizerParamGroup> paramgroups;
    if (name == "SGD")
    {
        auto pg2_options = torch::optim::SGDOptions(lr).momentum(momentum).nesterov(true)/*.weight_decay(0.0)*/;
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg2,
            std::make_unique<torch::optim::SGDOptions>(pg2_options)));

        auto pg0_options = torch::optim::SGDOptions(lr).weight_decay(decay);
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg0,
            std::make_unique<torch::optim::SGDOptions>(pg0_options)));

        auto pg1_options = torch::optim::SGDOptions(lr)/*.weight_decay(0.0)*/;
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg1,
            std::make_unique<torch::optim::SGDOptions>(pg1_options)));

        return std::make_shared<torch::optim::SGD>(paramgroups, pg2_options);
    }
    else if (name == "RMSProp")
    {
        auto pg2_options = torch::optim::RMSpropOptions(lr).momentum(momentum).weight_decay(0.0);
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg2,
            std::make_unique<torch::optim::RMSpropOptions>(pg2_options)));

        auto pg0_options = torch::optim::RMSpropOptions(lr).weight_decay(decay);
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg0,
            std::make_unique<torch::optim::RMSpropOptions>(pg0_options)));

        auto pg1_options = torch::optim::RMSpropOptions(lr).weight_decay(0.0);
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg1,
            std::make_unique<torch::optim::RMSpropOptions>(pg1_options)));

        return std::make_shared<torch::optim::RMSprop>(paramgroups, pg2_options);
    }
    else if (name == "AdamW")
    {
        auto pg2_options = torch::optim::AdamWOptions(lr).betas(std::make_tuple(momentum, 0.999)).weight_decay(0.0);
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg2,
            std::make_unique<torch::optim::AdamWOptions>(pg2_options)));

        auto pg0_options = torch::optim::AdamWOptions(lr).weight_decay(decay);
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg0,
            std::make_unique<torch::optim::AdamWOptions>(pg0_options)));

        auto pg1_options = torch::optim::AdamWOptions(lr).weight_decay(0.0);
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg1,
            std::make_unique<torch::optim::AdamWOptions>(pg1_options)));

        return std::make_shared<torch::optim::AdamW>(paramgroups, pg2_options);
    }
    else if (name == "Adam"){  // 可以作为if else分支default，确保总有一个有效值传回
        auto pg2_options = torch::optim::AdamOptions(lr).betas(std::make_tuple(momentum, 0.999)).weight_decay(0.0);
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg2,
            std::make_unique<torch::optim::AdamOptions>(pg2_options)));

        auto pg0_options = torch::optim::AdamOptions(lr).weight_decay(decay);
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg0,
            std::make_unique<torch::optim::AdamOptions>(pg0_options)));

        auto pg1_options = torch::optim::AdamOptions(lr).weight_decay(0.0);
        paramgroups.emplace_back(torch::optim::OptimizerParamGroup(pg1,
            std::make_unique<torch::optim::AdamOptions>(pg1_options)));

        return std::make_shared<torch::optim::Adam>(paramgroups, pg2_options);
    }
    else
        LOG(ERROR) << "Not support you set optimizer type: " << name;

    return nullptr;
}

void BaseTrainer::setupScheduler()
{
    bool is_linear = std::get<bool>(m_args["cos_lr"]) == false;
    float lrf = std::get<float>(m_args["lrf"]);
    int epochs = std::get<int>(m_args["epochs"]);

    CHECK(optimizer != nullptr) << "optimizer must be initialize before setup scheduler.";

    scheduler = std::make_shared<LambdaLR>(optimizer, 1.0f, lrf, epochs, is_linear);
    std::cout << ColorString("scheduler: ") << " lrf: " << lrf << " cos_lr: " << (is_linear? " False\n" : " True\n");
}

void BaseTrainer::resumeLastTraining()
{
    start_epochs = 0;
    epochs = std::get<int>(m_args["epochs"]);
}

void BaseTrainer::save_batch_sample_image(int epoch, int base_epoch, int start_epoch,
		VariantData& batch)
{
    if(m_bPlotsEnable && (epoch < (start_epochs + 3) || 
        (epoch >= base_epoch && epoch < (base_epoch + 3))))
    {
        torch::Tensor batch_idx = std::get<torch::Tensor>(batch[GET_VDK(VDK::BATCH_IDX)]);
        torch::Tensor cls = std::get<torch::Tensor>(batch[GET_VDK(VDK::CLS)]);
        torch::Tensor bboxes = std::get<torch::Tensor>(batch[GET_VDK(VDK::BBOXES)]);
        torch::Tensor img = std::get<torch::Tensor>(batch[GET_VDK(VDK::IMG)]);
        auto targets = torch::cat({ batch_idx.view({-1, 1}),
                                    cls.view({-1, 1}),
                                    bboxes }, 1);

        if(epoch < (start_epoch + 3)) 
            Singleton_PlotBatchImages::getInstance(m_saveDir)->push_data(img, targets, "batch_train%d", epoch - start_epoch);
        else 
            Singleton_PlotBatchImages::getInstance(m_saveDir)->push_data(img, targets, "batch_train%d"+std::to_string(base_epoch), epoch - start_epoch);
    }
}

void BaseTrainer::doWarmup(int epoch, int ni, int nw, int nbs, int batch_size)
{
    auto linear_interp = [](float x, const std::vector<float>& xp, const std::vector<float>& fp) {
        auto it = std::lower_bound(xp.begin(), xp.end(), x);
        if (it == xp.begin()) return fp[0];
        if (it == xp.end()) return fp.back();

        size_t xp_idx = it - xp.begin();    //在那两点间，用这两个坐标计算直线斜率，插值得到返回y坐标 
        float x0 = xp[xp_idx - 1], x1 = xp[xp_idx];
        float f0 = fp[xp_idx - 1], f1 = fp[xp_idx];
        return f0 + (x - x0) * (f1 - f0) / (x1 - x0);
        };

    std::vector<float> xi;
    xi.push_back(0.f);
    xi.push_back(float(nw));
    auto lr_lambda = create_lr_lambda(std::get<bool>(m_args["cos_lr"]), std::get<float>(m_args["lrf"]), epochs);

    accumulate = std::roundf(std::max(1.f, linear_interp(ni, xi, { 1.f, float(nbs / batch_size) })));

    for (size_t j = 0; j < optimizer->param_groups().size(); j++)
    {
        // # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
        float start_lr = (j == 0) ? std::get<float>(m_args["warmup_bias_lr"]) : 0.0f;
        float end_lr = std::get<float>(m_args["lr0"]) * lr_lambda(epoch);     // python原代码中x["initial_lr"] C++中没有找到
        std::vector<float> fp = { start_lr, end_lr };
        optimizer->param_groups()[j].options().set_lr(double(linear_interp(float(ni), xi, fp)));

        auto& options = optimizer->param_groups()[j].options();
        auto m = linear_interp(ni, xi, { std::get<float>(m_args["warmup_momentum"]),
                                        std::get<float>(m_args["momentum"])});

        if (auto sgd_options = dynamic_cast<torch::optim::SGDOptions*>(&options))
        {
            sgd_options->momentum(m);
        }
        else if (auto rmsprop_options = dynamic_cast<torch::optim::RMSpropOptions*>(&options))
        {
            rmsprop_options->momentum(m);
        }
    }
}

std::string BaseTrainer::refresh_batch_end_info(int epoch, int epochs, int idx, int num_targets)
{
    auto tmpstr = m_compute_loss->getLossitemsInfo(epoch, epochs, idx, num_targets);
    return tmpstr + std::string("  ");
}

void BaseTrainer::on_train_epoch_over(int epoch)
{   // 除了验证的动作之外的动作
    bool save_flag = train_loader->getTotalSamples() > 1000 ? true : false;    // 大数据集，每次都强制保存
    if (!save_flag && std::get<bool>(m_args["save"]))
    {
        int save_period = std::get<int>(m_args["save_period"]);
        if(save_period > 0 && ((epoch + 1) % save_period) == 0) 
            save_flag = true;
    }

    if (save_flag)
    {
        save_model(epoch);
    }

    // test code 

    specialTreatWhenEpochOver();
}

void BaseTrainer::on_train_over()
{
    specialTreatWhenTrainover();
    save_model(epochs);
}

void BaseTrainer::save_model(int epoch)
{
    torch::serialize::OutputArchive out_arch;

    c10::IValue i_epoch(epochs);
    out_arch.write("epoch", i_epoch);

    write_envsets_to_pt(out_arch);

    this->model->save_model(out_arch);

    out_arch.save_to(m_lastPtFile);
    
    std::cout << ColorString("Save model: ") << m_lastPtFile << " over." << std::endl;
}
