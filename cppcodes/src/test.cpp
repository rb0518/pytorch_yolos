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


void test(
    std::shared_ptr<Model> model,   // model or nullpytr
    std::string root_,          
    VariantConfigs opt,             // opt 
    std::vector<std::string> cls_names,
    std::shared_ptr<LoadImagesAndLabels> val_datasets, // if not val, set = nullptr
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
    std::shared_ptr<Model> model_tmp; 
    std::string save_dir = save_dir_;
    if(is_training)
    {
        model_ptr = model->get();
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
                        .append(std::get<std::string>(opt["project"]))
                        .append(std::get<std::string>(opt["name"])).string();
            save_dir = increment_path(strtmp);
        }
        std::cout << "save dir: " << save_dir << std::endl;

        auto cfg_file = std::filesystem::path(root_).append(std::get<std::string>(opt["cfg"])).string();
        std::cout << "cfg_file: " << cfg_file << std::endl;
        if(false == std::filesystem::exists(std::filesystem::path(cfg_file)))
        {
            LOG(ERROR) << "cfg_file: " << cfg_file << " not exist." ;
            return;
        }
        std::string weights_file = std::filesystem::path(root_).append(weights).string();
        if(false == std::filesystem::exists(weights_file))
        {
            LOG(ERROR) << "weights file: " << weights << " not exist." ;
            return;
        }
        model_tmp = std::make_shared<Model>(cfg_file, nc, imgsz, imgsz, 3, false);
        model_ptr = model_tmp->get();
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
            return;
        }
    }
    
    model_ptr->eval();
    if(false == is_training)
        std::cout << "Model create over..." << std::endl;
    torch::Tensor iouv = torch::linspace(0.5, 0.95, 10);
    iouv = iouv.to(device);
    auto niou = iouv.numel();

    std::shared_ptr<LoadImagesAndLabels> val_load_datasets=nullptr;
    if(val_datasets == nullptr)
    {
        std::cout << "val path: " << val_path << std::endl;
        VariantConfigs hyp = set_cfg_hyp_default();
        bool augment = false;
        bool rect = true;
        bool image_weights = false;
        bool cache_images = false;
        int gs = 32;
        float pad = 0.5f;
        val_load_datasets = std::make_shared<LoadImagesAndLabels>(val_path, hyp, imgsz, batch_size, augment,
            rect, image_weights, cache_images, nc == 1, gs, pad, "");
        std::cout << " create val datasets:  LoadImagesAndLabels over ...." << std::endl;
    }
    else
    {
        val_load_datasets = std::move(val_datasets);
    }
    auto test_datasets = val_load_datasets->map(CustomCollate());
    auto total_images = test_datasets.size();
    int num_images = 0;
    if (total_images.has_value())
        num_images = *total_images;
    //std::cout << "test dataset, total images: " << num_images << " batch_size: " << batch_size << std::endl;
    auto dataloader_options = torch::data::DataLoaderOptions().batch_size(batch_size).workers(std::get<int>(opt["workers"]));
    auto val_dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                std::move(test_datasets), dataloader_options);
    if(false == is_training)                
        std::cout << "init dataloader over..." << std::endl;
    if (is_training == false)
    {
        auto tmp_tenosr = torch::zeros({ 1, 3, imgsz, imgsz }).to(device).to(torch::kFloat);
        model_ptr->forward(tmp_tenosr);
    }

    int seen = 0;
    //auto confusion_matrix = ConfusionMatrix(nc);

    auto loss = torch::zeros({ 3 }).to(device);

    std::vector<std::vector<torch::Tensor>> stats;
    int batch_i = 0;
    progressbar pbar(num_images/batch_size, 36);
    for (auto batch : *val_dataloader)
    {
        auto img = batch.data;
        auto targets = batch.target;

        img = img.to(device).to(torch::kFloat);
        img = img / 255.f;
        targets = targets.to(device);

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
        torch::Tensor out;
        std::vector<torch::Tensor> train_out;
        std::tie(out, train_out) = model_ptr->forward(img);
        if (save_pred)
        {
            if (batch_i < 3)
            {
                std::string save_name = "batch_val" + std::to_string(batch_i) + "_label.jpg";
                plot_images(img.clone().to(torch::kCPU), targets.clone().to(torch::kCPU), save_dir, save_name, true, cls_names);
            }
        }
        //std::cout << "out: " << out.sizes() << std::endl;


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
        std::vector<torch::Tensor> out_vc = non_max_suppression(out, conf_thres, iou_thres, {}, 
            false, true, /*lb*/{});

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
}

