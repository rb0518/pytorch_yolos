#include "test.h"

#include <filesystem>

#include "yaml-cpp/yaml.h"

#include "datasets.h"
#include "metrics.h"
#include "progressbar.h"

#include "metrics.h"

torch::Tensor scale_coords(std::vector<float> img1_shape, torch::Tensor coords, std::vector<float> img0_shape,
    std::vector<std::vector<float>> ratio_pad = {})
{
    std::vector<float> pad;
    float gain;
    if (ratio_pad.size() == 0)
    {
        gain = std::min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[0]);
        pad.push_back((img1_shape[1] - img0_shape[1] * gain) / 2);
        pad.push_back((img1_shape[0] - img0_shape[0] * gain) / 2);
    }
    else
    {
        gain = ratio_pad[0][0];
        pad = ratio_pad[1];
    }

    coords.index_put_({ torch::indexing::Slice(), 0 },
        coords.index({ torch::indexing::Slice(), 0 }) - pad[0]);
    coords.index_put_({ torch::indexing::Slice(), 2 },
        coords.index({ torch::indexing::Slice(), 2 }) - pad[0]);
    coords.index_put_({ torch::indexing::Slice(), 1 },
        coords.index({ torch::indexing::Slice(), 1 }) - pad[1]);
    coords.index_put_({ torch::indexing::Slice(), 3 },
        coords.index({ torch::indexing::Slice(), 3 }) - pad[1]);

    return coords.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4)}).div(gain);
}


void test(
    std::shared_ptr<Model> model,
    std::string root_,
    VariantConfigs opt,
    std::string data_cfg,
    std::string weights /*= ""*/,
    int imgsz /*= 640*/,
    int batch_size/* = 32*/,
    float conf_thres/* = 0.001f*/,
    float iou_thres /*= 0.6f*/,
    std::shared_ptr<LoadImagesAndLabels> val_datasets
    )
{
    bool training = model != nullptr ? true : false;
    auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    std::string save_dir;

    if(training)
    {
        device = (*model)->parameters().begin()->device();
        //if (device.type() == torch::kCPU)
        //    std::cout << "device == torch::kCPU " << std::endl;
        //std::cout << "device == torch::kCUDA" << std::endl;
    }
    else
    {   //不是从train.cpp中调用，需要读取环境变量，初始化Model，并加载weights
        
        //save_dir = increment_path()
    }

    (*model)->eval();

    torch::Tensor iouv = torch::linspace(0.5, 0.95, 10);
    iouv = iouv.to(device);
    auto niou = iouv.numel();

    YAML::Node data_dict;
    std::filesystem::path data_yaml = std::filesystem::path(root_).append(data_cfg);
    if (std::filesystem::exists(data_yaml))
    {
        // ???后期需要进行专项修改，因为data yaml中有两种格式，采取map和list
        data_dict = YAML::LoadFile(data_yaml.string());
        //std::cout << "Load data file: " << data_cfg << std::endl;
    }
    else
    {
        LOG(ERROR) << "not found you offer data yaml file: " << data_cfg;
    }
    auto is_coco = data_cfg.substr(data_cfg.length() - 10, data_cfg.length() - 1) == "coco.yaml";
    auto nc = std::get<bool>(opt["single_cls"]) == true ? 1 : data_dict["nc"].as<int>();

    VariantConfigs hyp;
    std::string train_path;
    std::shared_ptr<LoadImagesAndLabels> val_load_datasets=nullptr;
    if(training == false)
    {
        bool augment = false;
        bool rect = true;
        bool image_weights = false;
        bool cache_images = false;
        int gs = 32;
        float pad = 0.5f;
        val_load_datasets = std::make_shared<LoadImagesAndLabels>(train_path, hyp, imgsz, batch_size, augment,
            rect, image_weights, cache_images, nc == 1, gs, pad, "");
        std::cout << " create val datasets:  LoadImagesAndLabels over ...." << std::endl;
    }
    else
    {
        val_load_datasets = std::move(val_datasets);
    }
    auto test_datasets = val_load_datasets->map(torch::data::transforms::Stack<>());
    auto total_images = test_datasets.size();
    int num_images = 0;
    if (total_images.has_value())
        num_images = *total_images;
    //std::cout << "test dataset, total images: " << num_images << " batch_size: " << batch_size << std::endl;
    auto dataloader_options = torch::data::DataLoaderOptions().batch_size(batch_size).workers(std::get<int>(opt["workers"]));
    auto val_dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                std::move(test_datasets), dataloader_options);
    //std::cout << "init dataloader over..." << std::endl;
    if (training == false)
    {
        auto tmp_tenosr = torch::zeros({ 1, 3, imgsz, imgsz }).to(device).to(torch::kFloat);
        (*model)->forward(tmp_tenosr);
    }

    int seen = 0;
    //auto confusion_matrix = ConfusionMatrix(nc);

    //float p = 0.f;
    //float r = 0.f;
    //float mp = 0.f;
    //float mr = 0.f;
    //float map50 = 0.f;
    //float map = 0.f;
    //float t0 = 0.f;
    //float t1 = 0.f;

    auto loss = torch::zeros({ 3 }).to(device);

//    std::vector<torch::Tensor> jdict, ap, ap_class;
    std::vector<std::vector<torch::Tensor>> stats;
    //std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<float>>> stats;
    int batch_i = 0;
    progressbar pbar(num_images/batch_size, 36);
    //std::cout << "start testing..." << std::endl;
    for (auto batch : *val_dataloader)
    {
        auto img = batch.data;
        auto targets = batch.target;
        if (img.sizes().size() == 5)
            img = img.squeeze(0);
        if (targets.sizes().size() == 3)
            targets = targets.squeeze(0);

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
        std::tie(out, train_out) = (*model)->forward(img);
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
            // auto mask = targets.index({torch::indexing::Slice(), 0}) == i;
            // auto selected = targets.index({mask, torch::indexing::Slice(1, torch::indexing::None)});
            lb.push_back(selected);
        }
        std::vector<torch::Tensor> out_vc = non_max_suppression(out, conf_thres, iou_thres, {}, 
            false, true, /*lb*/{});
        for (int i = 0; i < out_vc.size(); i++)
        {
            // 筛选出对应的labels;
            auto labels = lb[i];
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

            if(out_vc[i].size(0) != 0)
                std::cout << i << " out_vc: " << out_vc[i].size(0) << out_vc[i].sizes() << std::endl;

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
            auto r_scale_coords =scale_coords(std::vector<float>({ float(img[i].size(1)), float(img[i].size(2)) }),
                predn.index({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 4) }),
                std::vector<float>({ float(img[i].size(1)), float(img[i].size(2))}),
                std::vector<std::vector<float>>({{1.f, 1.f}, {0.f, 0.f}})
                );
                predn.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 4) }, r_scale_coords);
            std::cout << " predn :"<< predn.sizes() << std::endl;
            
            // Evaluate
            torch::Tensor correct;
            if (nl)
            {
                auto tbox = xywh2xyxy(labels.index({torch::indexing::Slice(),
                    torch::indexing::Slice(1, 5)}));
                //std::cout << nl << " tbox: "  << tbox.sizes() << std::endl;

                tbox = scale_coords(std::vector<float>({ float(img[i].size(1)), float(img[i].size(2)) }),
                    tbox,
                    std::vector<float>({float(img[i].size(1)), float(img[i].size(2))}),
                    std::vector<std::vector<float>>({ {1.f, 1.f}, {0.f, 0.f} })
                );

                auto labelsn = torch::cat({ labels.index({torch::indexing::Slice(),
                    torch::indexing::Slice(0, 1)}), tbox}, 1).to(device);
                predn = predn.to(device);
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

        // plot images
        if (batch_i < 3)
        {

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
    sprintf(temp_strs, "  all:%d %d mp: %8.4f mr: %8.4f map50 %8.4f map %8.4f",
        seen, nt.sum().item().toInt(), mp.item().toFloat(), mr.item().toFloat(),
        map50.item().toFloat(), map.item().toFloat());
    std::string result_str = std::string(temp_strs);
    std::cout << result_str << std::endl;
}

