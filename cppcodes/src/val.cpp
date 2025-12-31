#include "val.h"
#include "detect_data.h"

#include "yolo.h"
#include "nms.h"
#include "ops.h"
#include "general.h"
#include "metrics.h"

DetectValidator::DetectValidator(VariantConfigs& args_input, std::string root_dir, 
    std::shared_ptr<BaseModelImpl> model_ptr /*= nullptr*/,
    std::shared_ptr<BaseLoss> loss_ptr /*= nullptr*/)
	: BaseValidate(args_input, root_dir, model_ptr, loss_ptr)
{
}

void DetectValidator::readSpecialSettins()
{
    m_conf = getVariantValue<float>(&m_args, "conf", 0.01f);
    m_iouthreshold = getVariantValue<float>(&m_args, "iou", 0.7f);
}

std::shared_ptr<BaseModelImpl> DetectValidator::createModel(std::string model_cfg)
{
	return std::make_shared<ModelImpl>(model_cfg, 3, &m_args);
}

void DetectValidator::loadWeights(std::string model_cfg)
{
    torch::serialize::InputArchive ckpt_in;
    ckpt_in.load_from(model_cfg);
    m_modelPtr->load_weights(ckpt_in);
    std::cout << " over." << std::endl;
}

void DetectValidator::createDataloader()
{
	m_valLoader = std::dynamic_pointer_cast<BaseDataloader>(
			std::make_shared<DetectDataload>(m_rootDir, m_args, 32, true));
	m_valLoader->initDatas();
}

void DetectValidator::createSpecialOjects()
{
    auto iouv = torch::linspace(0.5f, 0.95f, 10).to(m_device);
    for (int i = 0; i < iouv.numel(); i++)
        m_iouv.push_back(iouv[i].item<float>());
    std::cout << "Validator m_iouv.size() " << m_iouv.size() << std::endl;
}

void DetectValidator::beforeEpoch()
{
    m_stats.clear();
}

void DetectValidator::preprocessBatch(VariantData& batch)
{
    bool non_blocking = m_device.type() == c10::DeviceType::CUDA;

    auto img = std::get<torch::Tensor>(batch[GET_VDK(VDK::IMG)]);
    img = img.to(torch::kFloat).div(255.f);
    img = img.to(m_device, non_blocking);
    batch[GET_VDK(VDK::IMG)] = img;

    batch[GET_VDK(VDK::CLS)] = std::get<torch::Tensor>(batch[GET_VDK(VDK::CLS)]).to(m_device, non_blocking);
    batch[GET_VDK(VDK::BATCH_IDX)] = std::get<torch::Tensor>(batch[GET_VDK(VDK::BATCH_IDX)]).to(m_device, non_blocking);
    batch[GET_VDK(VDK::BBOXES)] = std::get<torch::Tensor>(batch[GET_VDK(VDK::BBOXES)]).to(m_device, non_blocking);
}

std::vector<torch::Tensor> DetectValidator::postprocess(torch::Tensor preds)
{
    bool b_agnostic = getVariantValue<bool>(&m_args, "single_cls", false) || getVariantValue<bool>(&m_args, "agnostic_nms", false);
    int max_det = getVariantValue<int>(&m_args, "max_det", 300);

    auto outputs = non_max_suppression(
        preds,
        m_conf,
        m_iouthreshold,
        {},
        b_agnostic,
        !m_bSingle,
        {},
        max_det);

    return outputs;
}

void DetectValidator::update_metrics(std::vector<torch::Tensor> preds, VariantData batch)
{
    // self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    //std::cout << "update_metrics preds sizes() " << preds.size() << " " << preds[0].sizes() << std::endl;
    for (int si = 0; si < preds.size(); si++)
    {
        m_seen += 1;
        auto pred = preds[si];
        
        if(pred.size(0))
            std::cout << "si:" << si << " pred " << pred.sizes() << std::endl;

        int npr = pred.size(0);
        VariantData stat;

        stat["conf"] = torch::zeros(0, torch::TensorOptions().device(m_device));
        stat["pred_cls"] = torch::zeros(0, torch::TensorOptions().device(m_device));
        stat["tp"] = torch::zeros({npr, int64_t(m_iouv.size())}, torch::TensorOptions().device(m_device));

        VariantData pbatch = _prepare_batch(si, batch);
        torch::Tensor cls = std::get<torch::Tensor>(pbatch["cls"]);
        torch::Tensor bbox = std::get<torch::Tensor>(pbatch["bbox"]);

        int nl = cls.size(0);
        stat["target_cls"] = cls;
        auto [cls_matchs, cls_indices] = torch::_unique(cls, false, false);
        stat["target_img"] = cls_matchs;
        if (npr == 0)
        {
            if (nl) 
            {
                m_stats.emplace_back(stat);
            }
            continue;
        }

        if (m_bSingle)
            pred.index_put_({ torch::indexing::Slice(), 5 }, 0);

        auto predn = _prepare_pred(pred, pbatch);
        stat["conf"] = predn.index({ torch::indexing::Slice(), 4 }).clone();
        stat["pred_cls"] = predn.index({ torch::indexing::Slice(), 5 }).clone();

        if (nl)
        {
            stat["tp"] = _process_batch(predn, bbox, cls);
        }

        m_stats.emplace_back(stat);
    }
}

std::map<std::string, torch::Tensor> DetectValidator::get_stats()
{
    VariantData stats_cat = cat_stats();
    torch::Tensor target_cls = std::get<torch::Tensor>(stats_cat["target_cls"]).to(torch::kInt);
    torch::Tensor target_img = std::get<torch::Tensor>(stats_cat["target_img"]).to(torch::kInt);

    auto nt_per_class = torch::bincount(target_cls, {}, nc);
    auto nt_per_image = torch::bincount(target_img, {}, nc);
    torch::Tensor tp = std::get<torch::Tensor>(stats_cat["tp"]);
    torch::Tensor conf = std::get<torch::Tensor>(stats_cat["conf"]);
    torch::Tensor pred_cls = std::get<torch::Tensor>(stats_cat["pred_cls"]);
    target_cls = target_cls.to(torch::kFloat32);

    std::map<std::string, torch::Tensor> rets;
    if(stats_cat.size() && tp.any().item<bool>())
    {
        auto results = ap_per_class(tp, conf, pred_cls, target_cls);
        rets["p"] = std::get<2>(results);
        rets["r"] = std::get<3>(results);
        rets["ap"] = std::get<5>(results);
    }

    return rets;
}

VariantData DetectValidator::_prepare_batch(int si, VariantData batch)
{
    torch::Tensor idx = (std::get<torch::Tensor>(batch["batch_idx"]) == si).squeeze(-1);
    auto cls = std::get<torch::Tensor>(batch["cls"]).index({ idx }).squeeze(-1);
    auto bbox = std::get<torch::Tensor>(batch["bboxes"]).index({idx});

	// std::vector<std::string> keys;
	// std::transform(batch.begin(), batch.end(), std::back_inserter(keys), [](const auto& pair) {
	// 	return pair.first; });
	// std::for_each(keys.begin(), keys.end(), [](const auto& k) {std::cout << "key: " << k << std::endl; });


    std::tuple<float, float> ori_size = std::get<std::vector<std::tuple<float, float>>>(batch[GET_VDK(VDK::SRC_SIZE)])[si];
    std::tuple<float, float> imgsz = std::get < std::vector<std::tuple<float, float>>>(batch[GET_VDK(VDK::RESIZED)])[si];
    // std::tuple<float, float> pad = std::get<std::vector<std::tuple<float, float>>>(batch[GET_VDK(VDK::PAD)])[si];
    // std::tuple<float, float> ratio = std::get<std::vector<std::tuple<float, float>>>(batch[GET_VDK(VDK::RATIO)])[si];

    std::tuple<float, float> pad = std::make_tuple(std::get<0>(imgsz) - std::get<0>(ori_size), 
            std::get<1>(imgsz) - std::get<1>(ori_size));

    bbox = xywhn2xyxy(bbox, std::get<0>(ori_size), std::get<1>(ori_size), std::get<0>(pad), std::get<1>(pad));
    bbox.index({ torch::indexing::Slice(), 0 }).clip_(0.f, std::get<0>(imgsz));
    bbox.index({ torch::indexing::Slice(), 2 }).clip_(0.f, std::get<0>(imgsz));
    bbox.index({ torch::indexing::Slice(), 1 }).clip_(0.f, std::get<1>(imgsz));
    bbox.index({ torch::indexing::Slice(), 3 }).clip_(0.f, std::get<1>(imgsz));

    VariantData rets;
    rets["cls"] = cls;
    rets["bbox"] = bbox;
    rets["imgsz"] = std::vector<float>({ std::get<0>(imgsz), std::get<1>(imgsz) });
    return rets;
}

torch::Tensor DetectValidator::_prepare_pred(torch::Tensor pred, VariantData pbatch)
{
    torch::Tensor predn = pred.clone();
    std::vector<float> imgsz = std::get<std::vector<float>>(pbatch["imgsz"]);
    predn.index({ torch::indexing::Slice(), 0 }).clip_(0, imgsz[0]);
    predn.index({ torch::indexing::Slice(), 2 }).clip_(0, imgsz[0]);
    predn.index({ torch::indexing::Slice(), 1 }).clip_(0, imgsz[1]);
    predn.index({ torch::indexing::Slice(), 3 }).clip_(0, imgsz[1]);
    return predn;
}

torch::Tensor DetectValidator::_process_batch(torch::Tensor detections, torch::Tensor gt_bboxes, torch::Tensor gt_cls)
{
    torch::Tensor detections_bbox = detections.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4) });
    auto iou = box_iou(gt_bboxes, detections_bbox);

    auto pred_cls = detections.index({ torch::indexing::Slice(), 5 });
    auto rets = match_predictions(pred_cls, gt_cls, iou);
    return rets;
}

torch::Tensor DetectValidator::match_predictions(torch::Tensor pred_classes, torch::Tensor true_classes, torch::Tensor iou)
{
    torch::Tensor correct = torch::zeros({ pred_classes.size(0), int64_t(m_iouv.size()) }, torch::kBool);
    auto correct_class = true_classes.index({ torch::indexing::Slice(),torch::indexing::None }) == pred_classes;
    iou = iou * correct_class;

    iou = iou.cpu();
    //std::cout << iou.sizes() << std::endl;
    for (int i = 0; i < m_iouv.size(); i++)
    {
        float threshold = m_iouv[i];
        auto matches = torch::nonzero(iou >= threshold);
        if (matches.size(0) > 0)
        {
            if (matches.size(0) > 1)
            {
                // matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                auto iou_indices = iou.index({ 
                    matches.index({torch::indexing::Slice(), 0}), 
                    matches.index({torch::indexing::Slice(), 1}) }).argsort(-1, true);  // 倒序
                matches = matches.index({ iou_indices });
                // matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                torch::Tensor index_1;
                std::tie(std::ignore, index_1) = torch::_unique(matches.index({ torch::indexing::Slice(), 1 }));
                matches = matches.index({ index_1 });

                // matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                torch::Tensor index_0;
                std::tie(std::ignore, index_0) = torch::_unique(matches.index({ torch::indexing::Slice(), 0 }));
                matches = matches.index({ index_0 });
            }
            auto correct_index = matches.index({ torch::indexing::Slice(), 1 });
            correct_index = correct_index.to(torch::kInt64);
            correct.index_put_({ correct_index, i }, torch::ones({ correct_index.size(0) }, torch::kBool));
        }
    }
    correct = correct.to(pred_classes.device());
    return correct;
}

VariantData DetectValidator::cat_stats()
{
    std::vector<torch::Tensor> vt_tp;
    std::vector<torch::Tensor> vt_conf;
    std::vector<torch::Tensor> vt_pred_cls;
    std::vector<torch::Tensor> vt_target_cls;
    std::vector<torch::Tensor> vt_target_img;
    for (int i = 0; i < m_stats.size(); i++)
    {
        VariantData item = m_stats[i];
        vt_tp.push_back(std::get<torch::Tensor>(item["tp"]));
        vt_conf.push_back(std::get<torch::Tensor>(item["conf"]));
        vt_pred_cls.push_back(std::get<torch::Tensor>(item["pred_cls"]));
        vt_target_cls.push_back(std::get<torch::Tensor>(item["target_cls"]));
        vt_target_img.push_back(std::get<torch::Tensor>(item["target_img"]));

    }

    VariantData rets;
    rets["tp"] = torch::cat(vt_tp, 0);
    rets["conf"] = torch::cat(vt_conf, 0);
    rets["pred_cls"] = torch::cat(vt_pred_cls, 0);
    rets["target_cls"] = torch::cat(vt_target_cls, 0);  // coco128 929
    rets["target_img"] = torch::cat(vt_target_img, 0);  // coco128 362

    // std::cout << "tp: " << std::get<torch::Tensor>(rets["tp"]).sizes() << std::endl;
    // std::cout << "conf: " << std::get<torch::Tensor>(rets["conf"]).sizes() << std::endl;
    // std::cout << "pred_cls: " << std::get<torch::Tensor>(rets["pred_cls"]).sizes() << std::endl;
    // std::cout << "target_cls: " << std::get<torch::Tensor>(rets["target_cls"]).sizes() << std::endl;
    // std::cout << "target_img: " << std::get<torch::Tensor>(rets["target_img"]).sizes() << std::endl;

    m_instance = std::get<torch::Tensor>(rets["target_cls"]).size(0);
    return rets;
}

std::string DetectValidator::get_desc() 
{
    char temp_str[120];
    sprintf(temp_str, "%9s%7s%10s%10s%10s%10s%10s%10s ", " ", "class", "Image", "Instance", "Box(P", "R", "mAP50", "mAP(50-95)");
    return std::string(temp_str);
}

void DetectValidator::print_results(std::map<std::string, torch::Tensor> stats)
{
    char temp_str[120];
    std::string cls = m_bSingle ? "Single" : "All";
    int images = m_valLoader->getTotalSamples();

    torch::Tensor t_p = stats.count("p") ? stats["p"] : torch::zeros({1});
    torch::Tensor t_r = stats.count("r") ? stats["r"] : torch::zeros({1});
    torch::Tensor t_ap = stats.count("ap") ? stats["ap"] : torch::zeros({1, 10});

    //std::cout << "P: " << torch::mean(t_p) << " R " << torch::mean(t_r) << std::endl;
    float p = torch::mean(t_p).item<float>();
    float r = torch::mean(t_r).item<float>();

    auto m_t_ap = torch::mean(t_ap, 0);
    //std::cout << m_t_ap << std::endl;
    float mAP50 = m_t_ap[0].item<float>();
    float mAP = torch::mean(m_t_ap).item<float>(); 
    //std::cout << torch::mean(m_t_ap) << std::endl;

    sprintf(temp_str, "%9s%7s%10d%10d%10.5f%10.5f%10.5f%10.5f ", " ", cls.c_str(), images, m_instance, p, r, mAP50, mAP);
    std::cout << std::string(temp_str) << std::endl << std::endl;
}