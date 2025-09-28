#include <torch/torch.h>
#include <ATen/autocast_mode.h>     // autocast
#include <filesystem>
#include <tuple>
#include <fstream>

#include "train.h"
#include "utils.h"

#include "yolo.h"
#include "general.h"
#include "datasets.h"
#include "loss.h"
#include "lambda_lr.h"

#include "progressbar.h"
#include "plots.h"
#include "yaml_load.h"

#include "test.h"
#ifdef WIN32
void test_model(std::shared_ptr<Model> ptr_model, std::string strfilename = "../../data/images/bus.jpg");       // Working Directory = ${ProjectDir}
#else
void test_model(std::shared_ptr<Model> ptr_model, std::string strfilename = "../../data/images/bus.jpg");
#endif
std::tuple<std::string, std::string, std::string> create_dirs(const std::string save_dir)
{
    auto wdir = std::filesystem::path(save_dir).append("weights");
    if (!std::filesystem::exists(wdir))
        std::filesystem::create_directories(std::filesystem::path(wdir));

    auto last = std::filesystem::path(wdir).append("last.pt").string();
    auto best = std::filesystem::path(wdir).append("best.pt").string();
    auto results_file = std::filesystem::path(save_dir).append("results.txt").string();

    return std::make_tuple(last, best, results_file);
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

std::tuple<std::shared_ptr<torch::optim::Optimizer>, int, float> smart_optimizer(std::shared_ptr<Model> ptr_model, VariantConfigs& opt, VariantConfigs& hyp)
{
    auto model = ptr_model->get();
    const int nbs = 64;
    int batch_size = std::get<int>(opt["batch_size"]);
    int accumulate = std::max(static_cast<int>(std::round(nbs / batch_size)), 1);
    float lr0 = std::get<float>(hyp["lr0"]);
    float momentum = std::get<float>(hyp["momentum"]);
    float weight_decay = std::get<float>(hyp["weight_decay"]);
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

    if (std::get<bool>(opt["adam"]))
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

std::function<float(int)> create_lr_lambda(bool linear_lr, float lrf, int epochs) {
    if (linear_lr) {
        // 线性插值
        return [lrf, epochs](int x) -> float {
            return (1.0f - static_cast<float>(x) / (epochs - 1)) * (1.0f - lrf) + lrf;
            };
    }
    else {
        // 余弦插值
        return [lrf, epochs](int x) -> float {
            return ((1 - std::cos(x * M_PI / epochs)) / 2) * (lrf - 1) + 1;
            };
    }
}

void train(const std::string& _root, 
    VariantConfigs& hyp,
	VariantConfigs& opt,
    torch::Device& device,
    const std::string& jit_script_file
)
{
    show_cfg_info("opt", opt);
    show_cfg_info("hyp", hyp);

    std::cout << ColorString("Device: ", "Info") << device.type() << std::endl;
    auto save_dir = std::filesystem::path(_root).append(std::get<std::string>(opt["save_dir"])).string();
    std::cout << ColorString("Save dir: ", "Info") << save_dir << std::endl;
    auto epochs = std::get<int>(opt["epochs"]);
    auto batch_size = std::get<int>(opt["batch_size"]);
    auto total_batch_size = std::get<int>(opt["total_batch_size"]);
    auto weights = std::get<std::string>(opt["weights"]);

    if(std::filesystem::path(weights).stem().string() != "last")
    {   // optim和weight分开保存了，所以有多个pt文件
        weights = std::filesystem::path(weights).parent_path().append("last.pt").string();
        opt["weights"] = weights;
    }
    // Directories
    auto [last, best, results_file] = create_dirs(save_dir);
    std::cout << " last: " << last /*<< " best: " << best*/ << " results: " << results_file << std::endl;

    // Save run settings
    auto hyp_file = std::filesystem::path(save_dir).append("hyp.yaml").string();
    save_cfg_yaml(hyp, hyp_file);
    auto opt_file = std::filesystem::path(save_dir).append("opt.yaml").string();
    save_cfg_yaml(opt, opt_file);
    
    // Configure
    auto plots = std::get<bool>(opt["evolve"]) == false;
    auto cuda = device.type() == torch::kCUDA;
    init_torch_seek(2 + std::get<int>(opt["local_rank"])); // default init_seed(1)
    
    auto data_file = std::filesystem::path(_root).append(std::get<std::string>(opt["data"])).string();
    std::vector<std::string> names;
    std::string train_path;
    std::string val_path;
    if (std::filesystem::exists(std::filesystem::path(data_file)))
    {       
        read_data_yaml(data_file, train_path, val_path, names); // 对应两种yaml定义，list和map
    }
    else
    {
        LOG(ERROR) << "not found you offer data yaml file: " << data_file;
    }
    auto is_coco = data_file.substr(data_file.length() - 10, data_file.length() - 1) == "coco.yaml";
    auto nc = (std::get<bool>(opt["single_cls"])) ? 1 : names.size();

    if (std::get<bool>(opt["single_cls"]))
    {
        names.clear();
        names.emplace_back("item");
    }    

    if (names.size() != nc)
    {   // 不用glog CHECK或assert的原因是程序会因异常直接退出
        LOG(WARNING) << "names found in " << data_file << " items, but not equil the nc: " << nc << std::endl;
    }

    auto pretrained = std::filesystem::path(weights).extension().string() == ".pt"  && 
        std::filesystem::exists(std::filesystem::path(_root).append(weights));
    auto imgsz = std::get<std::vector<int>>(opt["img_size"])[0];//auto imgsz = 
    auto imgsz_test = std::get<std::vector<int>>(opt["img_size"])[1];

    auto cfg_file = std::get<std::string>(opt["cfg"]);
    if(cfg_file=="")
    {
        LOG(WARNING) << "cfg not define, use default sets yolov5s.yaml";
        cfg_file = std::filesystem::path(_root).append("models/yolov5s.yaml").string();
    }
    else
        cfg_file = std::filesystem::path(_root).append(cfg_file).string();

    std::shared_ptr<Model> ptr_model = std::make_shared<Model>(cfg_file, nc, imgsz, imgsz, 3, false); 
    auto model = ptr_model->get();
    model->show_modelinfo();
    model->to(device);

    bool resume_load_ok = false;
    int start_epoch = 0;
    int last_ni = 0;
    float last_lr = std::get<float>(hyp["lr0"]);
    torch::serialize::InputArchive ckpt_in;
    if(jit_script_file!= ""){
        std::cout << ColorString("Load pretrain weights: ", "Info") << jit_script_file << std::endl;
        if(std::filesystem::exists(std::filesystem::path(_root).append(jit_script_file)))
        {
            auto jit_file = std::filesystem::path(_root).append(jit_script_file).string();
            LoadWeightFromJitScript(jit_file, *model, true);
        }
        else{
            LOG(WARNING) << "load " << std::filesystem::path(_root).append(jit_script_file).string() << "not exist.";
        }
    }
    else if(std::get<bool>(opt["resume"]) && pretrained)
    {
        std::cout << ColorString("Load pretrain weights: ", "Info") << weights;
        std::string ckpt_path = std::filesystem::path(_root).append(weights)./*parent_path().append("ckpt.pt").*/string();
        ckpt_in.load_from(ckpt_path); // model optim epoch
        torch::serialize::InputArchive model_in;
        if (ckpt_in.try_read("model", model_in))
        {
            model->load(model_in);
            resume_load_ok = true;
            std::cout << " Over. " << std::endl;
        }
        if(!resume_load_ok) 
            std::cout << " wrong..." << std::endl;
    }
    else{
        std::cout << "\033[33m" << "Load pretrain weights: " << "\033[37m" << "None" << std::endl;
    }

    std::vector<std::string> freeze;    //原代码这里也没有处理初始化队列, 应该对应的是opt["freeze"]的层数
    for (auto p : model->named_parameters())
    {
        p.value().requires_grad_(true);

        if (std::count(freeze.begin(), freeze.end(), p.key()) > 0)
        {
            std::cout << "freezing " << p.key() << std::endl;
            p.value().requires_grad_(false);
        }
    }

    // Optimizer
    std::cout << ColorString("Optimizer: ", "Info");
    int accumulate = 1;
    float lr0 = 0.02;
    int nbs = 64;
    std::shared_ptr<torch::optim::Optimizer> optimizer{ nullptr };
    std::tie(optimizer, accumulate, lr0) = smart_optimizer(ptr_model, opt, hyp);
    // Resume
    auto best_fitness = 0.0;

    //auto ema = ModelEMA(ptr_model);
    if (resume_load_ok)
    {
        // optimizer
        torch::serialize::InputArchive optim_in;
        if (ckpt_in.try_read("optim", optim_in))
        {
            torch::Tensor tensor_size;
            if (optim_in.try_read("size", tensor_size)) // 区分是整个保存的，第一层是没有size的
            {
                int pg_size = tensor_size.item().toInt();
                if (pg_size == optimizer->param_groups().size())
                {
                    for (int pg_idx = 0; pg_idx < pg_size; pg_idx++)
                    {
                        std::string tmp_key = "param_groups/" + std::to_string(pg_idx);
                        torch::serialize::InputArchive options_tmp;
                        optim_in.read(tmp_key, options_tmp);
                        optimizer->param_groups()[pg_idx].options().serialize(options_tmp);
                    }
                }
            }
            else
            {
                optimizer->load(optim_in);
                std::cout << "optimizer->load() over... " << std::endl;
            }
            std::cout << std::endl;
        }

        // epochs
        torch::Tensor epoch_tensor;
        if (ckpt_in.try_read("epoch", epoch_tensor))
        {
            start_epoch = epoch_tensor.item().toInt() + 1;
            if (start_epoch >= epochs)
                epochs = start_epoch + epochs;
        }
    }

    auto lr_lambda = create_lr_lambda(std::get<bool>(opt["linear_lr"]), std::get<float>(hyp["lrf"]), epochs);
    auto scheduler = LambdaLR(optimizer, 1.0, std::get<float>(hyp["lrf"]), epochs, std::get<bool>(opt["linear_lr"]), start_epoch); 

    auto gs = std::max(32, model->get_stride_max()); // grid size (max stride)
    auto nl = model->last_module->nl;

    bool augment = true;
    bool rect = std::get<bool>(opt["rect"]);
    float pad = 0.0f;

    // create dataloader
    std::cout << ColorString("Train Datasets: ", "Info") << train_path << std::endl;
    auto [train_dataloaer, num_images] = create_dataloader(train_path, imgsz, nc, batch_size, gs,
                                                            opt, hyp, augment, pad); 
    
    std::cout << ColorString("Test Datasets: ", "Info") << val_path << std::endl;
    auto [val_dataloader, num_images_val] = create_dataloader(val_path, imgsz, nc, batch_size * 2, gs,
                                                            opt, hyp, augment, pad, true);
    std::cout << " create dataloader over ...." << std::endl;
    auto model_modules = model->last_module;

    // no opt.resume，输出tensorboard， 
    // no opt.noautoanchors 因为并没有支持catch，所以暂时也不完成

    hyp["box"] = std::get<float>(hyp["box"]) * float(3.f / nl);
    hyp["cls"] = std::get<float>(hyp["cls"]) * float((nc / 80.f) * (3.f / nl));
    hyp["obj"] = std::get<float>(hyp["obj"]) * float(pow(imgsz / 640.f, 2.f) * 3.f/nl);
    hyp["label_smoothing"] = std::get<float>(opt["label_smoothing"]);

    model->n_classes = nc;
    model->hyp = hyp;
    model->gr = 1.0;
    //model->class_weights =
    model->names = names;

    ComputeLoss compute_loss(model_modules, hyp);   // 一定要放在model->to(device)之后，不然程序中要自己判定传入、传出tensor在哪 ？
   
    auto nw = std::max(int(std::round(std::get<float>(hyp["warmup_epochs"]) * num_images)), 1000);

    int ni = 0;
    std::cout << "All init is over, start training..." << std::endl;    

    Singleton_PlotBatchImages::getInstance(save_dir, "batch_train")->reset_counter();
    for (auto epoch = start_epoch; epoch < epochs; epoch++)
    {
        model->to(device);
        model->train();

        if (std::get<bool>(opt["image_weights"]))
        {   //??? 暂时未完成 

        }

        int idx = 0;
        auto mloss = torch::zeros({4}).to(device).to(torch::kFloat);

        progressbar pbar(num_images/batch_size, 36);
        pbar.set_done_char(std::string("█"));

        optimizer->zero_grad();
        for (auto batch : *train_dataloaer)
        {
            torch::Tensor loss, loss_items;
            ni = idx + ((num_images / batch_size) + 1) * epoch;
           
            auto imgs = batch.data;
            auto target = batch.target;
            auto paths = batch.path;
            auto shape = batch.shape;
//            std::cout << "imgs: " << imgs.sizes() << "target: " << target.sizes() << std::endl;

            if(idx == 0 && epoch < (start_epoch + 3))
            {
                Singleton_PlotBatchImages::getInstance(save_dir, "batch_train")->push_data(imgs, target);
            }

            imgs = imgs.to(torch::kFloat) / 255.f;
            imgs = imgs.to(device);
  
            auto linear_interp = [](float x, const std::vector<float>& xp, const std::vector<float>& fp) {
                auto it = std::lower_bound(xp.begin(), xp.end(), x);
                if (it == xp.begin()) return fp[0];
                if (it == xp.end()) return fp.back();

                size_t xp_idx = it - xp.begin();    //在那两点间，用这两个坐标计算直线斜率，插值得到返回y坐标 
                float x0 = xp[xp_idx - 1], x1 = xp[xp_idx];
                float f0 = fp[xp_idx - 1], f1 = fp[xp_idx];
                return f0 + (x - x0) * (f1 - f0) / (x1 - x0);
                };
            
            // warmup
            if (ni <= nw )
            {
                std::vector<float> xi = { 0, float(nw) };

                accumulate = std::roundf(std::max(1.f, linear_interp(ni, xi, { 1.f, float(nbs / total_batch_size) })));
                //std::cout << "accumulate : " << accumulate << std::endl;
                for (size_t j = 0; j < optimizer->param_groups().size(); j++)
                {
                    float start_lr = (j == 2) ? std::get<float>(hyp["warmup_bias_lr"]) : 0.0f;
                    float end_lr = std::get<float>(hyp["lr0"]) * lr_lambda(epoch);
                    std::vector<float> fp = { start_lr, end_lr };
                    //std::cout << j <<" lr: " << optimizer->param_groups()[j].options().get_lr() << std::endl;
                    optimizer->param_groups()[j].options().set_lr(double(linear_interp(float(ni), xi, fp)));

                    //x.options().set_lr(double(linear_interp(float(ni), xi, fp)));
                    auto& options = optimizer->param_groups()[j].options();
                    double old_lr = optimizer->param_groups()[j].options().get_lr();
                    if (auto sgd_options = dynamic_cast<torch::optim::SGDOptions*>(&options))
                    {
                        //std::cout << "convert to SGD ok..." << std::endl;
                        auto m = linear_interp(ni, xi, {
                                    std::get<float>(hyp["warmup_momentum"]),
                                    std::get<float>(hyp["momentum"])
                            });
                        std::unique_ptr<torch::optim::SGDOptions> pg_options = std::make_unique<torch::optim::SGDOptions>(old_lr);
                        pg_options->momentum(m);
                        pg_options->nesterov(true);
                        optimizer->param_groups()[j].set_options(std::move(pg_options));
                    }
                    else if (auto adam_options = dynamic_cast<torch::optim::AdamOptions*>(&options))
                    {

                    }
                    else
                    {
                        // do nothing
                    }
                }
            }
           
            // multi-scale
            if (std::get<bool>(opt["multi_scale"]))
            {   // ???暂时未完成
                std::cout << "mulit-scale no support now." << std::endl;
            }

            {   // {}是来限定 set_autocast_enabled（） 的作用域范围
                at::autocast::set_autocast_enabled(torch::kCUDA, true);
                auto [zeros, pred, pred_mask] = model->forward(imgs);
                std::tie(loss, loss_items) = compute_loss(pred, target.to(device));
                if(std::get<bool>(opt["quad"]))
                    loss *= 4.f;
            }

            // loss = loss * 32.f;
            loss.backward();

            if (ni % accumulate == 0)
            {
                optimizer->step();
                optimizer->zero_grad();
                //ema.update(*model);
            }

            if(mloss.device().type() != loss_items.device().type())
            {
                mloss.to(loss_items.device());
            }
            mloss = (mloss * idx + loss_items) / float(idx + 1);

            std::stringstream ss;
            ss << epoch << "-" << epochs <<" " << idx << " b: " << std::to_string(mloss[0].item().toFloat());
            ss << " o: " << std::to_string(mloss[1].item().toFloat());
            ss << " c: " << std::to_string(mloss[2].item().toFloat());
            ss << " l: " << std::to_string(mloss[3].item().toFloat());
            ss << " t: " << std::to_string(target.size(0));

            idx+=1;
            pbar.set_prefix(ss.str() + "  ");
            pbar.update();
        }

        scheduler.step();
        std::cout << std::endl;
        if (false == std::get<bool>(opt["notest"]))
        {  // mAP
            bool save_flag = (epoch == start_epoch) ? true : false;
            test(ptr_model,//ema.ema_model_,
                _root,
                opt,
                names,
                std::move(val_dataloader),
                num_images_val,
                "",
                save_dir,
                "",
                80, 
                imgsz, 32, 0.2f, 0.45f,
                save_flag);
        }
        
        // 像coco.yaml数据集非常大，每轮训练都超过50分钏以上，如果只在最后一次保存，可能丢失训练时间太多了, 强制每次保存
        bool save_flag = num_images > 1000 ? true : false;
        if(false == save_flag)
        {
            int save_period = std::get<int>(opt["save_period"]);
            if( false == std::get<bool>(opt["nosave"]) && save_period > 0)
                if(((epoch + 1) % save_period) == 0)
                    save_flag = true;
        }

        if(save_flag){   
            save_checkpoint(last, model, optimizer, epoch);
        }
    } // end epochs

    save_checkpoint(last, model, optimizer, epochs-1);

    if (true == std::get<bool>(opt["notest"]))
    {  //
        auto weight_rel = std::filesystem::relative(last, std::filesystem::path(_root)).string();
        test(nullptr,//ema.ema_model_,
            _root,
            opt,
            names,
            std::move(val_dataloader),
            num_images_val,
            "",
            save_dir,
            weight_rel,
            80, 
            imgsz, 32, 0.2f, 0.45f, true
            );
    }
    else
        test_model(ptr_model);  // 测试代码
}

void test_model(std::shared_ptr<Model> ptr_model, std::string strfilename /*= "../../data/images/bus.jpg"*/)
{
    auto model = ptr_model->get();
    float confidence_threshold = 0.2f;
    float iou_threshold = 0.45f;
    int nms_max_bbox_size_ = 4096;  //只保留排前的多少个

    model->eval();
    torch::Device device = model->parameters().begin()->device();
    cv::Mat src_image = cv::imread(strfilename);
    cv::resize(src_image, src_image, cv::Size(640, 640));
    cv::Mat input_image;
    cv::cvtColor(src_image, input_image, cv::COLOR_BGR2RGB);
    
    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);
    at::Tensor input_tensor = torch::from_blob(input_image.data,
        { 1, input_image.rows, input_image.cols, 3 }).to(device);    // 补齐batch_size维
    input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).contiguous();

    auto [output_tensor, output_tensor_vec, output_tensor_pred_seg] = model->forward(input_tensor);// .toTuple()->elements()[0].toTensor();
    std::cout << "model forward return: " << output_tensor.sizes() << std::endl;
    // [1, 25200, 85]
    std::vector<torch::Tensor> bboxs;
    bboxs= non_max_suppression(output_tensor, confidence_threshold, iou_threshold, {},
        false, false, {});
    for(int i = 0; i < bboxs.size(); i++)
    {
        auto boxs = bboxs[i];
        std::cout << i << " box size: " << boxs.sizes() << std::endl;
        if(boxs.size(0)!=0)
        {
            for(int j = 0; j < boxs.size(0); j++)
            {
                auto x1 = boxs[j][0].item().toFloat();
                auto y1 = boxs[j][1].item().toFloat();
                auto x2 = boxs[j][2].item().toFloat();
                auto y2 = boxs[j][3].item().toFloat();
                auto score = boxs[j][4].item().toFloat();
                auto cls_id = boxs[j][5].item().toInt();

                auto typecolor = SingletonColors::getInstance()->get_color_scalar(cls_id);

                cv::rectangle(src_image, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)), typecolor, 2);
                std::cout << i <<" " << j <<" " << x1 << " " <<y1 << " " << x2 << " " << y2 << " cls " << cls_id << " s: " << score << std::endl;
            }

            cv::imshow("result", src_image);
            cv::waitKey();
            cv::destroyAllWindows();
        }
    }
}