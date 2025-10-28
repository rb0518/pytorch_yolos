#include "test.h"

#include <filesystem>

#include "yaml-cpp/yaml.h"

#include "datasets.h"
#include "metrics.h"
#include "progressbar.h"
#include "general.h"
#include "plots.h"

#include "metrics.h"

#include <regex>
//    std::regex pattern("yolov5([nsmlx])?.yaml");  return std::resgex_search(filename, pattern);
bool contains(const std::string& source, const std::string& to_find) {
    std::regex re(to_find);
    return std::regex_search(source, re);
}

torch::Tensor scale_coords(std::vector<float> img1_shape, torch::Tensor coords, std::vector<float> img0_shape,
    std::vector<std::vector<float>> ratio_pad = {})
{
    std::vector<float> pad;
    float gain;
    if (ratio_pad.size() == 0)
    {
        gain = std::min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]);
        pad.push_back((img1_shape[1] - img0_shape[1] * gain) / 2);
        pad.push_back((img1_shape[0] - img0_shape[0] * gain) / 2);
    }
    else
    {
        gain = ratio_pad[0][0];
        pad = ratio_pad[1];
    }
    //std::cout << "gain: " << gain << " pad: " << pad[0] << " " << pad[1] << std::endl;
    auto ret = torch::empty_like(coords);
    ret.index_put_({ torch::indexing::Slice(), 0 },
        coords.index({ torch::indexing::Slice(), 0 }) - pad[0]);
    ret.index_put_({ torch::indexing::Slice(), 2 },
        coords.index({ torch::indexing::Slice(), 2 }) - pad[0]);
    ret.index_put_({ torch::indexing::Slice(), 1 },
        coords.index({ torch::indexing::Slice(), 1 }) - pad[1]);
    ret.index_put_({ torch::indexing::Slice(), 3 },
        coords.index({ torch::indexing::Slice(), 3 }) - pad[1]);

    return ret = ret / gain;
}


Dataloader_Custom test(
    std::shared_ptr<ModelImpl> model,   // model or nullpytr
    std::string root_,          
    VariantConfigs args,             // default.yaml 
    std::vector<std::string> cls_names,
    Dataloader_Custom val_dataloader, // if not val, set = nullptr
    int val_total_number,
    std::string val_path,
    std::string save_dir_,
    std::string weights /*= ""*/,
    int nc/* = 80*/,
    int imgsz/* =640*/,
    int batch_size/*=32*/,
    float conf_thres/* = 0.001f*/,
    float iou_thres/* = 0.6f*/,
    bool save_pred /* = false*/
)
{
    bool is_training = model != nullptr ? true : false;
    auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    ModelImpl* model_ptr;
    std::shared_ptr<ModelImpl> model_tmp; 
    std::string save_dir = save_dir_;
    if(is_training)
    {
        model_ptr = model.get();
        device = model_ptr->parameters().begin()->device();

        //if (device.type() == torch::kCPU)
        //    std::cout << "device == torch::kCPU " << std::endl;
        //std::cout << "device == torch::kCUDA" << std::endl;
    }
    else
    {   //不是从train.cpp中调用，需要读取环境变量，初始化Model，并加载weights
        if(false == std::filesystem::exists(std::filesystem::path(save_dir)))
        {
            std::string strtmp = std::filesystem::path(root_)
                        .append(std::get<std::string>(args["project"]))
                        .append(std::get<std::string>(args["name"])).string();
            save_dir = increment_path(strtmp);
        }
        std::cout << "save dir: " << save_dir << std::endl;

        auto cfg_file = std::filesystem::path(root_).append(std::get<std::string>(args["model"])).string();
        std::cout << "cfg_file: " << cfg_file << std::endl;
        if(false == std::filesystem::exists(std::filesystem::path(cfg_file)))
        {
            LOG(ERROR) << "cfg_file: " << cfg_file << " not exist." ;
            return nullptr;
        }
        std::string weights_file = std::filesystem::path(root_).append(weights).string();
        if(false == std::filesystem::exists(weights_file))
        {
            LOG(ERROR) << "weights file: " << weights << " not exist." ;
            return nullptr;
        }
        model_tmp = std::make_shared<ModelImpl>(cfg_file, nc, imgsz, imgsz, 3, false);
        model_ptr = model_tmp.get();
        torch::serialize::InputArchive ckpt_in;
        ckpt_in.load_from(weights_file); // model optim epoch
        torch::serialize::InputArchive model_in;
        if (ckpt_in.try_read("model", model_in))
        {
            model_ptr->load(model_in);
            std::cout << "load " << weights_file << " over..." << std::endl;
        }
        else
        {
            LOG(ERROR) << "*** Load weight error! ";
            return nullptr;
        }
    }
    bool is_segment = model_ptr->is_segment;
    model_ptr->eval();
    if(false == is_training)
        std::cout << "Model create over..." << std::endl;
    torch::Tensor iouv = torch::linspace(0.5, 0.95, 10);
    iouv = iouv.to(device);
    auto niou = iouv.numel();

    int num_images = 0;
    
    if(val_dataloader == nullptr)
    {
        std::cout << "val path: " << val_path << std::endl;
        std::tie(val_dataloader, num_images) = create_dataloader(val_path, args,
                                                                32, true, is_segment);  // default follow: false, 4
    }
    else{
        num_images = val_total_number;
    }
    if(false == is_training)    
    {            
        std::cout << "init dataloader over..." << std::endl;
        auto tmp_tenosr = torch::zeros({ 1, 3, imgsz, imgsz }).to(device).to(torch::kFloat);
        model_ptr->forward(tmp_tenosr);
    }

    int seen = 0;
    auto loss = torch::zeros({ 3 }).to(device);

    std::vector<std::vector<torch::Tensor>> stats;
    int batch_i = 0;
    progressbar pbar(num_images/batch_size, 36);
    for (auto batch : *val_dataloader)
    {
        auto img = batch.at("img");
        auto bboxes = batch.at("bboxes");
        auto clses = batch.at("cls");
        auto batch_idx = batch.at("batch_idx");

        auto targets = torch::zeros({bboxes.size(0), 6}, 
            bboxes.options());

        targets.index_put_({"...", 0}, batch_idx);
        targets.index_put_({"...", 1}, clses);
        targets.index_put_({"...", torch::indexing::Slice(2, torch::indexing::None)},
                bboxes);
        torch::Tensor masks;
        if(batch.contains("mask"))
            masks = batch.at("mask");

        int nb = img.size(0);
        int height = img.size(2);
        int width = img.size(3);

        // 从[n, 6] ==> [m, 5] no with image idx
        auto get_onebatchlabels = [&](torch::Tensor targets, int idx) {
            // 获取当前批次si对应的标签数据
            auto mask = targets.index({torch::indexing::Slice(), 0}) == idx;
            auto selected = targets.index({mask, torch::indexing::Slice(1, torch::indexing::None)});
            return selected;
            };

        
        torch::NoGradGuard nograd;
        torch::Tensor preds;
        std::vector<torch::Tensor> train_out;
        torch::Tensor protos;
        std::tie(preds, train_out, protos) = model_ptr->forward(img);
        if (save_pred)
        {
            if (batch_i < 3)
            {
                std::string save_name = "batch_val" + std::to_string(batch_i) + "_label.jpg";
                plot_images(img.clone().to(torch::kCPU), targets.clone().to(torch::kCPU), save_dir, save_name, true, cls_names);
            }
        }
        //std::cout << "preds: " << preds.sizes() <<" protos: " << protos.sizes() << std::endl;


        // Compute loss
        // auto [tmp_loss, tmp_lossitems]= compute_loss(train_out, targets);
        // loss += tmp_lossitems.index({torch::indexing::Slice(0, 3)});
        // run NMS
        targets.index_put_(
            { torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None) },
            targets.index({ torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None) })
            .mul(torch::tensor({ width, height, width, height }).to(device))
            );
        //  lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        std::vector<torch::Tensor> lb;
        for(int i = 0; i < img.size(0); i++)
        {
            auto selected = get_onebatchlabels(targets, i);
            lb.push_back(selected);
        }

        int number_mask = is_segment ? 32 : 0;  // 后续直接从模型中读取，  ，
        std::vector<torch::Tensor> out_vc = non_max_suppression(preds, conf_thres, iou_thres, {}, 
            false, true, /*lb*/{}, 300, number_mask);

        if (save_pred)
        {
            if (batch_i < 3)
            {
                std::string save_name = "batch_val" + std::to_string(batch_i) + "_pred.jpg";
                plot_images_pred(img.clone().to(torch::kCPU), out_vc, save_dir, save_name, true, cls_names);
            }
        }

        for (int i = 0; i < out_vc.size(); i++)
        {
            // 筛选出对应的labels;
            auto labels = lb[i].to(device);
            auto nl = labels.size(0);
            std::vector<float> tcls;
            if (nl) {
                auto col = labels.index({ torch::indexing::Slice(), 0 });
                tcls.reserve(nl);
                for (int i = 0; i < nl; ++i) {
                    tcls.push_back(col[i].item<float>());
                }
            }
            seen += 1;

            //if(out_vc[i].size(0) != 0)
            //    std::cout << i << " out_vc: " << out_vc[i].size(0) << out_vc[i].sizes() << std::endl;

            if (out_vc[i].size(0) == 0)
            {
                if (nl)
                {
                    stats.push_back({ torch::zeros({ 0, niou }, torch::dtype(torch::kBool)),
                        torch::empty({0}),
                        torch::empty({0}),
                        torch::tensor(tcls)});
                }
                continue;
            }

            // predictions
            //if (single_cls)
            //    out_vc[i].index_put_({ torch::indexing::Slice(), 5 }, 0);
            auto predn = out_vc[i].clone().to(device);
            // std::cout << predn.dtype() << predn.device().type() << " " << std::endl;
            auto r_scale_coords = scale_coords(std::vector<float>({ float(width), float(height)}),
                predn.index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 4) }),
                std::vector<float>({ float(width), float(height)}),
                std::vector<std::vector<float>>({{1.f, 1.f}, {0.f, 0.f}})
                );
                predn.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 4) }, r_scale_coords);
            //std::cout << " predn :"<< predn.sizes() << std::endl;
            
            // Evaluate
            torch::Tensor correct;
            if (nl)
            {
                auto tbox = xywh2xyxy(labels.index({torch::indexing::Slice(),
                    torch::indexing::Slice(1, 5)})).to(device);
                //std::cout << nl << " tbox: "  << tbox.sizes() << std::endl;

                auto tbox_scale = scale_coords(std::vector<float>({float(width), float(height)}),
                    tbox,
                    std::vector<float>({float(width), float(height)}),
                    std::vector<std::vector<float>>({ {1.f, 1.f}, {0.f, 0.f} })
                );
                tbox = tbox_scale;
                //std::cout << "tobx: " << tbox.dtype() << " " << tbox.device().type() << std::endl;
                //std::cout << "labels: " << labels.dtype() << " " << labels.device().type() << std::endl;
                auto labelsn = torch::cat({ labels.index({torch::indexing::Slice(),
                    torch::indexing::Slice(0, 1)}), tbox}, 1).to(device);
                predn = predn.to(device);
                //std::cout << labelsn.sizes() << std::endl;
                correct = process_batch(predn, labelsn, iouv);
                //if (plots)
                //    confusion_matrix.process_batch(predn, labelsn);
            }
            else
            {
                correct = torch::zeros({ out_vc[i].size(0), niou }, torch::dtype(torch::kBool)).to(device);
            }
            stats.push_back({
                correct.to(torch::kCPU),
                out_vc[i].index({torch::indexing::Slice(), 4}).to(torch::kCPU),
                out_vc[i].index({torch::indexing::Slice(), 5}).to(torch::kCPU),
                torch::tensor(tcls)
                });

            // std::cout << correct.sizes() << " "
            //     << out_vc[i].index({ torch::indexing::Slice(), 4 }).sizes()
            //     << out_vc[i].index({ torch::indexing::Slice(), 5 }).sizes()
            //     << tcls.size() << std::endl;

            // Save/log
        }

        batch_i += 1;
        pbar.update();
    }
    auto compute_stats = [&](std::vector<std::vector<torch::Tensor>> stats) {
        std::vector<torch::Tensor> result;

        // 转置stats结构
        size_t num_metrics = stats[0].size();
        for (size_t i = 0; i < num_metrics; ++i) {
            std::vector<torch::Tensor> metric_tensors;

            // 收集当前metric的所有batch结果
            for (auto& batch : stats) {
                metric_tensors.push_back(batch[i]);
            }

            // 沿第0轴拼接
            result.push_back(torch::cat(metric_tensors, 0));
        }

        return result;
        };

    auto stats_total = compute_stats(stats);
    torch::Tensor nt = torch::zeros({ 1 });
    torch::Tensor mp = torch::zeros({ 1 });
    torch::Tensor mr = torch::zeros({ 1 });
    torch::Tensor map50 = torch::zeros({ 1 });
    torch::Tensor map = torch::zeros({ 1 });
    if (stats.size())
    {
        auto [p, r, ap, f1, ap_class] = ap_per_class(stats_total[0], stats_total[1], stats_total[2], stats_total[3]);
        auto ap50 = ap.index({ torch::indexing::Slice(), 0 });
        auto ap_mean = ap.mean(1);
        mp = p.mean();
        mr = r.mean();
        map50 = ap50.mean();
        map = ap_mean.mean();
        nt = torch::bincount(stats_total[3].to(torch::kInt64),
            /*weights=*/torch::Tensor(),
            /*minlength=*/nc);
        //std::cout << "after ap_per_class " << nt.sizes() << " map50: " << map50.sizes() << " map " << map.sizes() << std::endl;
    }
    char temp_strs[250];
    sprintf(temp_strs, "   All: %d %d mP: %7.5f mR: %7.5f mAP50 %7.5f mAP %7.5f",
        seen, nt.sum().item().toInt(), mp.item().toFloat(), mr.item().toFloat(),
        map50.item().toFloat(), map.item().toFloat());
    std::string result_str = std::string(temp_strs);
    std::cout << result_str << std::endl;

    return std::move(val_dataloader);
}

BaseValidator::BaseValidator(std::shared_ptr<Model> _model_ptr, VariantConfigs _args_input, std::string _root_dir)
    : model_ptr(_model_ptr), args(_args_input), root_dir(_root_dir)
{
    is_training = model_ptr != nullptr; // 未传入trainer中的model，则说明是要直接重新生成个model来验证

    setup_model();
    load_weight();

    init_device();
    init_dirs();

    init_dataloader();
}

void BaseValidator::init_dirs()
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

    task_name = std::get<std::string>(args["task"]);
    if (is_training)
    {   // 如果是训练中验证，沿用save_dir
        save_dir = std::filesystem::path(root_dir).append(std::get<std::string>(args["save_dir"])).string();
    }
    else
    {
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
    }
}

void BaseValidator::load_weight()
{
    if (is_training) 
        return;

    std::string weights_set = std::get<std::string>(args["weights"]);
    std::string weights_file = std::filesystem::path(root_dir).append(weights_set).string();
    if (false == std::filesystem::exists(weights_file))
    {
        LOG(ERROR) << "weights file: " << weights_set << " not exist.";
        return;
    }
    torch::serialize::InputArchive ckpt_in;
    ckpt_in.load_from(weights_file); // model optim epoch
    torch::serialize::InputArchive model_in;
    if (ckpt_in.try_read("model", model_in))
    {
        model_ptr->ptr()->load(model_in);
        std::cout << "load " << weights_file << " over..." << std::endl;
    }
    else
        LOG(ERROR) << "*** Load weight error! ";
}

void BaseValidator::setup_model()
{
    if (false == is_training)
    {
        auto tmp_path = std::filesystem::path(root_dir).append(std::get<std::string>(args["model"]));
        if (!std::filesystem::exists(tmp_path))
        {
            std::cout << ColorString("ERROR: ", "R") << "model yaml file not exists: " << tmp_path.string() << std::endl;
            return;
        }
        
        auto cfg_file = tmp_path.string();
        imgsz = std::get<int>(args["imgsz"]);
        YAML::Node cfgs = YAML::LoadFile(cfg_file);
        nc = cfgs["nc"].as<int>();
        model_ptr = std::make_shared<Model>(cfg_file, nc, imgsz, imgsz, 3);
        if (model_ptr == nullptr)
        {
            LOG(ERROR) << "model init error.\n";
        }
    }
}

void BaseValidator::init_device()
{
    if (is_training)
    {
        device = model_ptr->get()->parameters().begin()->device();
        return;
    }

    std::cout << "input device type: " << std::get<std::string>(args["device"]) << "\n";
    device = get_device(std::get<std::string>(args["device"]));
    std::cout << ColorString("Device: ") << device.type() << std::endl;
}

void BaseValidator::init_dataloader()
{
    if (task_name == "segment")
        val_loader = new DataloaderBase(root_dir, args, 32, true, true);
    else
        val_loader = new DataloaderBase(root_dir, args, 32, true, false);
}

YoloCustomExample BaseValidator::preprocess(YoloCustomExample& batch)
{
    bool non_blocking = this->device.type() == c10::DeviceType::CUDA;
    
    for (auto& item : batch)
    {
        batch.at(item.key()) = item.value().to(this->device, non_blocking);
    }
    batch.at("img") = batch.at("img").to(torch::kFloat) / 255.f;

    return batch;
}

std::vector<torch::Dict<std::string, torch::Tensor>> BaseValidator::postprocess(torch::Tensor preds)
{
    std::string tmp = std::get<std::string>(args["conf"]);
    float conf = 0.25f;
    if(tmp != "")
        conf = std::get<float>(args["conf"]);

    auto outputs = ops::non_max_suppression(
        preds,
        conf,
        std::get<float>(args["iou"]),
        {},
        (std::get<bool>(args["single_cls"]) || std::get<bool>(args["agnostic_nms"])),
        this->nc > 1,
        {},
        std::get<int>(args["max_det"])
        );

    std::vector<torch::Dict<std::string, torch::Tensor>> rets;
    for (int i = 0; i < outputs.size(); i++)
    {
        torch::Dict<std::string, torch::Tensor> one_pred_out;

        torch::Tensor out = outputs[i];
        int size_1 = out.size(1);

        auto out_split = out.split({ 4, 1, 1, size_1 - 6 }, 1);
        std::cout << "out_split size: " << out_split.size() << " " << out_split[0].sizes() << std::endl;
        one_pred_out.insert("bboxes", out_split[0]);
        one_pred_out.insert("conf", out_split[1]);
        one_pred_out.insert("cls", out_split[2]);
        one_pred_out.insert("extra", out_split[3]);

        rets.push_back(one_pred_out);
    }

    return rets;
}

void BaseValidator::update_metrics(std::vector<torch::Dict<std::string, torch::Tensor>> preds,
    torch::Dict<std::string, torch::Tensor> batch)
{
    int batch_count = preds.size();
    for (int si = 0; si < batch_count; si++)
    {
        this->seen += 1;

        auto pbatch = this->_prepare_batch(si, batch);
        auto predn = this->_prepare_pred(preds[si]);
    }
}


void BaseValidator::do_validate()
{
    _model_eval();

    int nb = val_loader->get_total_samples(true);
    
    progressbar pbar(nb, 30);
    pbar.set_done_char(std::string("█"));
    int idx_epoch = 0;
    this->seen = 0;
    for (auto batch : *(val_loader->dataloader))
    {
        if (idx_epoch < 3)
        {
            // save_batch_sample_image(batch, idx_epoch);
        }
        auto prepro_batch = preprocess(batch);
        int num_targets = prepro_batch.at("bboxes").size(0);

        // 1 Inference
        auto [pred_t, train_v, mask_t] = model_ptr->get()->forward(prepro_batch.at("img"));

        // 2 Loss 暂时未修改，后续要将v8DetectionLoss集成到model中，并添加loss函数

        // 3 Postprocess
        auto preds = postprocess(pred_t);

        // 4 update_metrics(preds, prepro_batch)


        if (std::get<bool>(args["plots"]) && idx_epoch < 3)
        {

        }


        idx_epoch++;
        pbar.update();
    }
}

void BaseValidator::_model_eval()
{
    this->model_ptr->get()->eval();
}

/*
    """
    Prepare a batch of images and annotations for validation.

    Args:
        si (int): Batch index.
        batch (dict[str, Any]): Batch data containing images and annotations.

    Returns:
        (dict[str, Any]): Prepared batch with processed annotations.
    """
*/
torch::Dict<std::string, torch::Tensor> BaseValidator::_prepare_batch(int si, 
    torch::Dict<std::string, torch::Tensor> batch)
{
    auto idx = batch.at("batch_idx") == si;
    auto cls = batch.at("cls").index({ idx }).squeeze(-1);
    auto bbox = batch.at("bboxes").index({ idx });
    /*
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        batch和target中坐标都是包括偏移的，暂时不考虑
    */

    if (cls.size(0)) // 有标注
    {
        bbox = xywh2xyxy(bbox);
        torch::Tensor scale = torch::tensor({ 640,640,640,640 },
            torch::TensorOptions().device(this->device));
        bbox = bbox * scale;
    }
    torch::Dict<std::string, torch::Tensor> ret;
    ret.insert("cls", cls);
    ret.insert("bboxes", bbox);
    
    return ret;        
}

torch::Dict<std::string, torch::Tensor> BaseValidator::_prepare_pred(torch::Dict<std::string, torch::Tensor> pred)
{
    // if(std::get<bool>(args["single_cls"])
    //      pred.at("cls") = pred.at("cls") * 0;
    return pred;
}
