
#include <ATen/autocast_mode.h>     // autocast
#include <filesystem>
#include <tuple>
#include <fstream>
#include <cmath>

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
#include "torch_utils.h"

#include "augmentations.h"

#include "test.h"
#ifdef WIN32

//void test_model(std::shared_ptr<Model> ptr_model, std::string strfilename = "../../data/images/zidane.jpg");       // Working Directory = ${ProjectDir}
void test_model(std::shared_ptr<ModelImpl> ptr_model, std::string strfilename = "../../data/images/bus.jpg");       // Working Directory = ${ProjectDir}
#else
//void test_model(std::shared_ptr<Model> ptr_model, std::string strfilename = "../../data/images/zidane.jpg");
void test_model(std::shared_ptr<ModelImpl> ptr_model, std::string strfilename = "../../data/images/bus.jpg");
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

void test_model(std::shared_ptr<ModelImpl> ptr_model, std::string strfilename /*= "../../data/images/bus.jpg"*/)
{
    auto model = ptr_model.get();
    float confidence_threshold = 0.2f;
    float iou_threshold = 0.45f;
    int nms_max_bbox_size_ = 4096;  //只保留排前的多少个

    model->eval();

    bool is_segment = model->is_segment;
    torch::Device device = model->parameters().begin()->device();
    cv::Mat src_image = cv::imread(strfilename);

    std::vector<float> ratio, pad;
    int img_size = 640;
    std::tie(src_image, ratio, pad)= letterbox(src_image, std::make_pair(img_size, img_size),
			cv::Scalar(114, 114, 114), false, false, false, 32);

    cv::Mat input_image;
    cv::cvtColor(src_image, input_image, cv::COLOR_BGR2RGB);
    
    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);
    at::Tensor input_tensor = torch::from_blob(input_image.data,
        { 1, input_image.rows, input_image.cols, 3 }).to(device);    // 补齐batch_size维
    input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).contiguous();

    auto [preds, output_tensor_vec, protos] = model->forward(input_tensor);// .toTuple()->elements()[0].toTensor();

    std::cout << "model forward return size: " << preds.sizes() << " protos: " << protos.sizes() << std::endl;
    std::cout << "model forward return type: " << preds.dtype() << " protos: " << protos.dtype() << std::endl;
    // [1, 25200, 85]  // new 1, 84, 8400
    int nm = 0;
    if(is_segment)  nm = 32;

    int bs = preds.size(0); // batch_size
    int nc = model->n_classes;
    if(nm != (preds.size(1)- nc - 4))
    {
        LOG(ERROR) << "preds size(1) != nc + 4 + nm ";
        return;
    }

    auto bboxs= ops::non_max_suppression(preds);
    std::vector<cv::Mat> mask_overlays;
    for(int i = 0; i < bboxs.size(); i++)
    {
        auto boxs = bboxs[i];
        std::cout << i << " box size: " << boxs.sizes() << " type: " << boxs.dtype() << std::endl;
        if(boxs.size(0)!=0)
        {
            torch::Tensor masks;            
            if(is_segment)
            {
                auto proto = protos[i];
                proto = proto.to(boxs.device());
                auto mask_in = boxs.index({torch::indexing::Slice(), torch::indexing::Slice(6, 6 + nm)});
                auto bboxes = boxs.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)});
                std::vector<int64_t> shape = {input_image.rows, input_image.cols};
                masks = process_mask(proto, mask_in, bboxes, shape, true);
                std::cout << "is_segment " << boxs.sizes() << " masks: " << masks.sizes() << " " << masks.dtype() << " " << sizeof(torch::kFloat32) << std::endl;
                cv::split(src_image, mask_overlays);

            }

            // 修改为倒序，置信度高的mask数据后处理
            for(int j = boxs.size(0)-1; j >= 0; j--)
            {
                auto x1 = boxs[j][0].item().toFloat();
                auto y1 = boxs[j][1].item().toFloat();
                auto x2 = boxs[j][2].item().toFloat();
                auto y2 = boxs[j][3].item().toFloat();
                auto score = boxs[j][4].item().toFloat();
                auto cls_id = boxs[j][5].item().toInt();

                auto typecolor = SingletonColors::getInstance()->get_color_scalar(cls_id);

                cv::rectangle(src_image, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)), typecolor, 2);
                char info_s[250];
                sprintf(info_s, "%3d %3d box: [ %8.2f, %8.2f - %8.2f , %8.2f ] cls: %-3d score: %6.4f", i, j, 
                    x1, y1, x2, y2, cls_id, score);
                std::cout << std::string(info_s) << std::endl;

                if(is_segment)
                {
                    auto mask_b = masks[j].clone().unsqueeze(0);

                    cv::Mat mat_mask = cv::Mat(mask_b.size(1), mask_b.size(2), CV_32FC1);
                    std::memcpy((void*)mat_mask.data, mask_b.data_ptr(), mask_b.element_size()*mask_b.numel());
                    cv::imshow("seg_"+std::to_string(j), mat_mask);

                    auto [c_r, c_g, c_b] = SingletonColors::getInstance()->get_color_uchars(cls_id);
                    float f_mask_thr = 0.5 /*+ std::max(0.0f, (0.5f - score))/10.f*/; // 二值化
                    mask_b = mask_b.gt_(f_mask_thr);    
                    mask_b = mask_b.to(torch::kByte);
                    int c = mask_b.size(0);
                    int h = mask_b.size(1);
                    int w = mask_b.size(2);

                    float f_r = float(c_r);
                    float f_g = float(c_g);
                    float f_b = float(c_b);
                    uchar* mask_b_ptr = (uchar*)mask_b.data_ptr();
                    uchar* over_r = (uchar*)mask_overlays[2].data;
                    uchar* over_g = (uchar*)mask_overlays[1].data;
                    uchar* over_b = (uchar*)mask_overlays[0].data;

                    for (int k_pixel = 0; k_pixel < mask_b.numel(); k_pixel++) 
                    {
                        float f_front = float(*mask_b_ptr) * 0.4f;
                        float f_back = 1.f - f_front;

                        float s_r = float(*over_r);
                        float s_g = float(*over_g);
                        float s_b = float(*over_b);

                        float f_r_tmp = s_r * f_back + f_r * f_front;
                        float f_g_tmp = s_g * f_back + f_g * f_front;
                        float f_b_tmp = s_b * f_back + f_b * f_front;

                        *over_r = uchar(f_r_tmp);
                        *over_g = uchar(f_g_tmp);
                        *over_b = uchar(f_b_tmp);


                        mask_b_ptr += 1;
                        over_r += 1;
                        over_g += 1;
                        over_b += 1;
                    }

                    
                    // auto m_f = cv::Mat(h, w, CV_8UC1);
                    
                    // std::cout << "h * w: " << (h*w) << " numel: " << mask_b.numel() << " " << sizeof(CV_8UC1) << " " << mask_b.element_size() << std::endl;
                    // std::memcpy((void*)m_f.data, mask_b.data_ptr(), mask_b.element_size() * mask_b.numel());
                    // cv::imshow("seg"+std::to_string(j), m_f);
//                    cv::imwrite("seg" + std::to_string(j) + ".jpg", mat_mask);
                }
            }

            cv::imshow("result", src_image);

            if (is_segment)
            {
                cv::Mat segment_result;
                cv::merge(mask_overlays, segment_result);
                cv::imshow("seg_result", segment_result);
            }

            cv::waitKey();
            cv::destroyAllWindows();
        }
    }
}

// ------------------------  start BaseTrain ------------------
BaseTrainer::BaseTrainer(std::string _root, VariantConfigs cfg_default)
    : root_dir(_root), args(cfg_default)
{
    show_cfg_info("BaseTrainer cfg: ", args);
    init_device();

    //init_torch_seek(std::get<int>(args["seed"]));
    task_name = std::get<std::string>(args["task"]);
    std::cout << ColorString("Task: ") << task_name << std::endl;

    init_dirs();
    setup_model();
    load_pretrained();

    init_dataloader();

    setup_optimizer();
    setup_scheduler();

    resume_training();
}

void BaseTrainer::init_device()
{
    std::cout << "input device type: " << std::get<std::string>(args["device"]) << "\n";
    device = get_device(std::get<std::string>(args["device"]));
    std::cout << ColorString("Device: ") << device.type() << std::endl;
}

void BaseTrainer::init_dirs() 
{
    auto get_task_suffix = [](const std::string& task_name, std::string& project) {
        if (task_name == "detect") { // do nothing
        }
        else if (task_name == "segment")
            project = project + "_seg";
        else if (task_name == "pose")
            project = project + "_pose";
        else if (task_name == "obb")
            project = project + "_obb";
        else if (task_name == "classify")
            project = project + "_class";
        else
            LOG(WARNING) << "you input task " << task_name << "not support.";
        };

    std::string project = std::get<std::string>(args["project"]);
    if (project == "")
    {
        LOG(WARNING) << "please input project name use --project. use default runs/train";
        project = "runs/train";
        get_task_suffix(task_name, project);
    }
    std::string name = std::get<std::string>(args["name"]);
    if (name == "")
    {
        LOG(WARNING) << "please input project name use --name. use default runs/train";
        name = "exp";
    }
    auto save_dir_path = std::filesystem::path(root_dir).append(project).append(name);
    if (std::filesystem::exists(save_dir_path))
    {
        if (std::get<bool>(args["exist_ok"]))
        {

            args["project"] = project;
            args["name"] = name;
            args["save_dir"] = save_dir_path.string();
        }
        else
        {
            auto prj_and_name = increment_path(save_dir_path.string(), false);
            name = std::filesystem::path(prj_and_name).filename().string();
            args["project"] = project;
            args["name"] = name;
            args["save_dir"] = prj_and_name;
        }
    }
    else
    {
        args["project"] = project;
        args["name"] = name;
        args["save_dir"] = save_dir_path.string();
    }
    save_dir = std::get<std::string>(args["save_dir"]);
    auto wdir_path = std::filesystem::path(save_dir).append("weights");
    if (!std::filesystem::exists(wdir_path))
        std::filesystem::create_directories(std::filesystem::path(wdir_path));
    last_pt_file = wdir_path.append("last.pt").string();
    best_pt_file = wdir_path.append("best.pt").string();
    results_file = std::filesystem::path(save_dir).append("results.txt").string();

    std::cout << ColorString("save dir: ", "info") << save_dir << "\n";
    // 转换为相对路径
    args["save_dir"] = std::filesystem::relative(std::filesystem::path(save_dir), 
                                            std::filesystem::path(root_dir)).string();

}

void BaseTrainer::setup_model()
{
    auto tmp_cfg_file_path = std::filesystem::path(root_dir).append(std::get<std::string>(args["model"]));
    std::string cfg_file = "";
    if (std::filesystem::exists(tmp_cfg_file_path))
        cfg_file = tmp_cfg_file_path.string();
    else
    {
        LOG(ERROR) << "set model cfg yaml file: " << tmp_cfg_file_path.string() << "not exists.";
        exit(-1);
    }
    auto imgsz = std::get<int>(args["imgsz"]);
    model = Model(cfg_file, 80, imgsz, imgsz, 3);
    if (model.is_empty())
    {
        LOG(ERROR) << "Create model error.";
        exit(-1);
    }

    model->set_args_prt(&args); // 传递指针，适应全局变量需要

    model->show_modelinfo();
    model->to(device);
}

void BaseTrainer::load_pretrained()
{
    bool load_from_jit = false;
    std::string jit_weights_file = std::get<std::string>(args["jit_weights"]);
    if (jit_weights_file != "")
    {
        auto path = std::filesystem::path(root_dir).append(jit_weights_file);
        if (std::filesystem::exists(path))
        {
            load_from_jit = true;
            jit_weights_file = path.string();

            LoadWeightFromJitScript(jit_weights_file, *model, true);
        }
    }

    if (std::get<bool>(args["resume"]) && !load_cfg_yaml)
    {

    }
}

void BaseTrainer::freeze_modules()
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

void BaseTrainer::init_dataloader()
{
    if(task_name == "segment")
        train_loader = new DataloaderBase(root_dir, args, 32, false, true);
    else
        train_loader = new DataloaderBase(root_dir, args, 32, false, false);
}

void BaseTrainer::setup_optimizer()
{
    int nbs = std::get<int>(args["nbs"]);
    int batch_size = train_loader->get_batch_size();
    accumulate = std::max(int(std::roundf(float(nbs) / float(batch_size))), 1);

    float weight_decay = std::get<float>(args["weight_decay"]);
    std::cout << "read weigth_decay : " << weight_decay << " accumulate " << accumulate << " batch_size: " << batch_size  << std::endl;
    weight_decay = weight_decay * float(batch_size) * float(accumulate) / float(nbs);
    int iterations = std::ceil(train_loader->get_total_samples() / std::max(batch_size, nbs))* std::get<int>(args["epochs"]);

    std::string name = std::get<std::string>(args["optimizer"]);
    float lr = std::get<float>(args["lr0"]);
    float momentum = std::get<float>(args["momentum"]);

    std::cout << " new weigth_decay : " << weight_decay << " iterations: " << iterations << "name: " << name << std::endl;


    build_optiomizer(model->n_classes, name, lr, momentum, weight_decay, iterations);
    if(name == "auto")
        args["warmup_bias_lr"] = 0.0f;
}

void BaseTrainer::build_optiomizer(int nc,std::string optim_name, 
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

    if (name == "SGD")
    {
        std::cout << "SGD" << " lr0 " << lr << std::endl;
        optimizer = std::make_shared<torch::optim::SGD>(
            pg2, torch::optim::SGDOptions(lr)
            .momentum(momentum)
            .nesterov(true).weight_decay(0.0));

        if (!pg0.empty()) {
            std::unique_ptr<torch::optim::SGDOptions> pg0_options = std::make_unique<torch::optim::SGDOptions>(lr);
            pg0_options->weight_decay(decay);
            torch::optim::OptimizerParamGroup paramgroup_pg0(pg0);
            paramgroup_pg0.set_options(std::move(pg0_options));

            optimizer->add_param_group(paramgroup_pg0);
        }
        if (!pg1.empty()) {
            std::unique_ptr<torch::optim::SGDOptions> pg1_options = std::make_unique<torch::optim::SGDOptions>(lr);
            pg1_options->weight_decay(0.0);
            torch::optim::OptimizerParamGroup paramgroup_pg1(pg1);
            paramgroup_pg1.set_options(std::move(pg1_options));

            optimizer->add_param_group(paramgroup_pg1);
        }
    }
    else if (name == "RMSProp")
    {
        std::cout << "SGD" << " lr0 " << lr << std::endl;
        optimizer = std::make_shared<torch::optim::RMSprop>(
            pg2, torch::optim::RMSpropOptions(lr)
            .momentum(momentum).weight_decay(0.0));

        if (!pg0.empty()) {
            std::unique_ptr<torch::optim::RMSpropOptions> pg0_options = std::make_unique<torch::optim::RMSpropOptions>(lr);
            pg0_options->weight_decay(decay);
            torch::optim::OptimizerParamGroup paramgroup_pg0(pg0);
            paramgroup_pg0.set_options(std::move(pg0_options));

            optimizer->add_param_group(paramgroup_pg0);
        }
        if (!pg1.empty()) {
            std::unique_ptr<torch::optim::RMSpropOptions> pg1_options = std::make_unique<torch::optim::RMSpropOptions>(lr);
            pg1_options->weight_decay(0.0);
            torch::optim::OptimizerParamGroup paramgroup_pg1(pg1);
            paramgroup_pg1.set_options(std::move(pg1_options));
            optimizer->add_param_group(paramgroup_pg1);
        }
    }
    else if (name == "AdamW")
    {
        std::cout << "AdamW" << " lr0 " << lr << std::endl;
        auto adamw_option = torch::optim::AdamWOptions(lr).betas(std::make_tuple(momentum, 0.999)).weight_decay(0.0);
        optimizer = std::make_shared<torch::optim::AdamW>(
            pg2, adamw_option);

        if (!pg0.empty()) {
            std::unique_ptr<torch::optim::AdamWOptions> pg0_options = std::make_unique<torch::optim::AdamWOptions>(lr);
            pg0_options->weight_decay(decay);
            torch::optim::OptimizerParamGroup paramgroup_pg0(pg0);
            paramgroup_pg0.set_options(std::move(pg0_options));

            optimizer->add_param_group(paramgroup_pg0);
        }

        if (!pg1.empty()) {
            std::unique_ptr<torch::optim::AdamWOptions> pg1_options = std::make_unique<torch::optim::AdamWOptions>(lr);
            pg1_options->weight_decay(0.0);
            torch::optim::OptimizerParamGroup paramgroup_pg1(pg1);
            paramgroup_pg1.set_options(std::move(pg1_options));
            optimizer->add_param_group(paramgroup_pg1);
        }
    }
    else {  // 其它情况:Adam, Adamax, NAdam RAdam在optim中没有，均用Adam
        name = "Adam";
        std::cout << "Adam" << " lr0 " << lr << std::endl;
        auto adam_option = torch::optim::AdamOptions(lr).betas(std::make_tuple(momentum, 0.999)).weight_decay(0.0);
        optimizer = std::make_shared<torch::optim::Adam>(
            pg2, adam_option);

        if (!pg0.empty()) {
            std::unique_ptr<torch::optim::AdamOptions> pg0_options = std::make_unique<torch::optim::AdamOptions>(lr);
            pg0_options->weight_decay(decay);
            torch::optim::OptimizerParamGroup paramgroup_pg0(pg0);
            paramgroup_pg0.set_options(std::move(pg0_options));

            optimizer->add_param_group(paramgroup_pg0);
        }

        if (!pg1.empty()) {
            std::unique_ptr<torch::optim::AdamOptions> pg1_options = std::make_unique<torch::optim::AdamOptions>(lr);
            pg1_options->weight_decay(0.0);
            torch::optim::OptimizerParamGroup paramgroup_pg1(pg1);
            paramgroup_pg1.set_options(std::move(pg1_options));
            optimizer->add_param_group(paramgroup_pg1);
        }
    }
    optimizer_name = name;
    std::cout << ColorString("optimizer: ", "info") << name << " (lr = " << lr
        << " momentum: " << momentum << ") with parameter groups g[1]: "
        << pg1.size() << " weight(decay=0.0) g[0]: " << pg0.size() << " weight(decay = "
        << decay << " ) g[2]: " << pg2.size() << " bias(decay=0.0)" << std::endl;
}

void BaseTrainer::setup_scheduler()
{
    bool is_linear = std::get<bool>(args["cos_lr"]) != true;
    float lrf = std::get<float>(args["lrf"]);
    int epochs = std::get<int>(args["epochs"]);
    scheduler = std::make_shared<LambdaLR>(optimizer, 1.0f, lrf, epochs, is_linear);

    std::cout << ColorString("scheduler: ") << " lrf: " << lrf << " cos_lr: " << (is_linear? " False\n" : " True\n");
}

void BaseTrainer::resume_training()
{
    start_epochs = 0;
    epochs = std::get<int>(args["epochs"]);
}

void BaseTrainer::save_batch_sample_image(int epoch, int base_epoch, int start_epoch,
		torch::Tensor data, torch::Tensor targets)
{
    if(epoch >= start_epoch && epoch < (start_epoch + 3))
    {   
        //std::cout << "save as start_epoch sample. \n"; 
        Singleton_PlotBatchImages::getInstance(save_dir)->push_data(data, targets, "batch_train", epoch - start_epoch);
    }
    else 
    {
        if(epoch >= base_epoch && epoch < (base_epoch + 3))
        {
            //std::cout << "save as start_epoch force no mosaic. \n" ;
            Singleton_PlotBatchImages::getInstance(save_dir)->push_data(data, targets, "batch_train"+std::to_string(base_epoch), epoch - start_epoch);
        }
    }
}


void BaseTrainer::do_train()
{
    int nb = train_loader->get_total_samples(true); // number of batchs
    int nw = -1;
    int nbs = std::get<int>(args["nbs"]);
    int warmup_epochs = std::get<float>(args["warmup_epochs"]);
    if (warmup_epochs > 0)
        nw = std::max(warmup_epochs * nb, 100); // 

    int base_epochs = epochs - std::get<int>(args["close_mosaic"]);
    optimizer->zero_grad();
    int ni = 0;
    std::shared_ptr<v8DetectionLossImpl> compute_loss = std::make_shared<v8DetectionLossImpl>(model.ptr());

    std::cout << ColorString("Start training:") << ", start_epoch: " << start_epochs <<" end epochs: " << epochs << " number of batch: " << nb << std::endl;

    for (int epoch = start_epochs; epoch < epochs; epoch++)
    {
        train_loader->cloase_mosaic(epoch >= base_epochs);       

        scheduler->step();
        model->train();

        progressbar pbar(nb, 30);
        pbar.set_done_char(std::string("█"));
        
        // 记录box, cls, dfl 三项loss
        auto mloss = torch::zeros({ 3 }).to(device).to(torch::kFloat);

        int idx_epoch = 0;
        int last_opt_step = -1;

        for (auto batch : *(train_loader->dataloader))
        {
            ni = idx_epoch + nb * epoch;

            if (ni <= nw)
                do_warmup(epoch, ni, nw, nbs, train_loader->get_batch_size());
            #ifdef _DEBUG_FOR_EXPORT_IMPORT_TENSORS_
            if(std::get<bool>(args["use_unified_batch"]))
            {
                batch.insert("img", load_tensordata_from_file("test_img.pt"));
                batch.insert("batch_idx", load_tensordata_from_file("test_batch_idx.pt"));
                batch.insert("cls", load_tensordata_from_file("test_cls.pt"));
                batch.insert("bboxes", load_tensordata_from_file("test_bboxes.pt"));
                std::cout << "use unified batch data over.\n";
            }
            #endif
            auto prepro_data = preprocess_batch(batch);

            int num_targets = prepro_data[1].size(0);
            if(idx_epoch == 0)
                save_batch_sample_image(epoch, base_epochs, start_epochs, prepro_data[0], prepro_data[1]);
            
            torch::Tensor loss, loss_items;

            {   // {}是来限定 set_autocast_enabled（） 的作用域范围
                at::autocast::set_autocast_enabled(torch::kCUDA, true);
                auto [pred_t, train_v, mask_t] = model->forward(prepro_data[0].to(device));

                std::tie(loss, loss_items) = compute_loss->forward(train_v, batch);
            }
            loss.backward();

            if((ni - last_opt_step) > accumulate)
            {
                optimizer->step();
                optimizer->zero_grad();
                last_opt_step = ni;
            }

            pbar.set_prefix(refresh_batch_end_info(epoch, epochs, idx_epoch, num_targets, mloss, loss_items));
            idx_epoch+=1;
            pbar.update();
        }
        scheduler->step();
        std::cout << std::endl;

        on_train_epoch_over(epoch);

        if (std::get<bool>(args["val"]) || epochs == (epoch + 1))    // # (bool) validate/test during training
        {   // add call valitate class do_valitate at here

        }
    }
}

std::vector<torch::Tensor> BaseTrainer::preprocess_batch(YoloCustomExample& batch)
{
    auto imgs = batch.at("img").to(torch::kFloat);
    imgs.div_(255.f);

    auto bboxes = batch.at("bboxes");
    auto clses = batch.at("cls");
    auto batch_idx = batch.at("batch_idx");
    auto targets = torch::zeros({ bboxes.size(0), 6 },
        bboxes.options());

    targets.index_put_({ "...", 0 }, batch_idx);
    targets.index_put_({ "...", 1 }, clses);
    targets.index_put_({ "...", torch::indexing::Slice(2, torch::indexing::None) },
        bboxes);
    return { imgs, targets };
}

void BaseTrainer::do_warmup(int epoch, int ni, int nw, int nbs, int batch_size)
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
    auto lr_lambda = create_lr_lambda(std::get<bool>(args["cos_lr"]), std::get<float>(args["lrf"]), epochs);

    accumulate = std::roundf(std::max(1.f, linear_interp(ni, xi, { 1.f, float(nbs / batch_size) })));

    for (size_t j = 0; j < optimizer->param_groups().size(); j++)
    {
        float start_lr = (j == 2) ? std::get<float>(args["warmup_bias_lr"]) : 0.0f;
        float end_lr = std::get<float>(args["lr0"]) * lr_lambda(epoch);     // python原代码中x["initial_lr"] C++中没有找到
        std::vector<float> fp = { start_lr, end_lr };
        optimizer->param_groups()[j].options().set_lr(double(linear_interp(float(ni), xi, fp)));

        //x.options().set_lr(double(linear_interp(float(ni), xi, fp)));
        auto& options = optimizer->param_groups()[j].options();
        double old_lr = optimizer->param_groups()[j].options().get_lr();

        if (j == 0)     // 初始化时只有这个是设置了的momentum
        {
            auto m = linear_interp(ni, xi, {
            std::get<float>(args["warmup_momentum"]),
            std::get<float>(args["momentum"])
                });

            if (auto sgd_options = dynamic_cast<torch::optim::SGDOptions*>(&options))
            {
                sgd_options->momentum(m);
            }
            else if (auto adamw_options = dynamic_cast<torch::optim::AdamWOptions*>(&options))
            {
                adamw_options->betas(std::make_tuple(m, 0.999));
            }
            else if (auto adam_options = dynamic_cast<torch::optim::AdamOptions*>(&options))
            {
                adam_options->betas(std::make_tuple(m, 0.999));
            }
            else if (auto rmsprop_options = dynamic_cast<torch::optim::RMSpropOptions*>(&options))
            {
                rmsprop_options->momentum(m);
            }
        }
    }
}

std::string BaseTrainer::refresh_batch_end_info(int epoch, int epochs, int idx, int num_targets,
		torch::Tensor& mloss, torch::Tensor& loss_items)
{
    mloss.to(loss_items.device());
    mloss = (mloss * idx + loss_items) / float(idx + 1);
    char temp_cstr[120];
    sprintf(temp_cstr, "%4d-%-4d%7d%15.5f%15.5f%15.5f%6d", (epoch+1), epochs, idx,
        mloss[0].item().toFloat(),
        mloss[1].item().toFloat(),
        mloss[2].item().toFloat(),
        num_targets);
    return std::string(temp_cstr) + std::string("  ");
}

void BaseTrainer::on_train_epoch_over(int epoch)
{   // 除了验证的动作之外的动作
    bool save_flag = this->train_loader->get_total_samples(false) > 1000 ? true : false;    // 大数据集，每次都强制保存
    if (!save_flag && std::get<bool>(args["save"]))
    {
        int save_period = std::get<int>(args["save_period"]);
        if(((epoch + 1) % save_period) == 0 || (epoch+1) == this->epochs) save_flag = true;
    }

    if (save_flag)
    {
        save_model();
    }

    // test code 
    if((epoch+1) == this->epochs)
    {
        test_model(model.ptr());
    }
}
