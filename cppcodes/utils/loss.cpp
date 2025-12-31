#include "loss.h"
#include "general.h"
#include "utils.h"
#include "ops.h"
//  c10::Dict<std::string, torch::IValue> hyp;
#include <torch/torch.h>
#include <torch/script.h>
#include <cmath>
#include <filesystem>

//--------------------------- start V8DetectionLoss --------------------------
v8DetectionLossImpl::v8DetectionLossImpl(std::shared_ptr<BaseModelImpl> model, torch::Device _device, int tal_topk/* = 10*/)
    : BaseLoss()
{
    this->device = _device;

    bce = torch::nn::BCEWithLogitsLoss(
        torch::nn::BCEWithLogitsLossOptions().reduction(torch::kNone));

    hyp = model->p_args;
    // check
    if(hyp == nullptr || hyp->size() == 0)
    {
        std::cout << ColorString("v8DetectionLossImpl", "R") << "get model::p_args is nullptr" << std::endl;
    }

    dfl_ratio = getVariantValue(hyp, "dfl", 1.0f);
    box_ratio = getVariantValue(hyp, "box", 1.0f);
    cls_ratio = getVariantValue(hyp, "cls", 1.0f);

    this->stride = model->get_stride();
    this->nc = model->n_classes;
    this->reg_max = model->get_regmax();

    std::cout << "v8DetectionLossImpl stride " << this->stride << " nc: " << this->nc << " reg_max: " << reg_max << std::endl;

    this->no = this->nc + this->reg_max * 4;
    this->use_dfl = this->reg_max > 1;
    //topk = tal_topk, num_classes = self.nc, alpha = 0.5, beta = 6.0
    assigner = TaskAlignedAssigner(tal_topk, nc, 0.5f, 6.0f);
    bbox_loss = BboxLoss(reg_max);
    bbox_loss->to(device);
    this->proj = torch::arange(reg_max, torch::TensorOptions().dtype(torch::kFloat).device(device));
}

torch::Tensor v8DetectionLossImpl::preprocess(torch::Tensor targets,
    int batch_size, torch::Tensor scale_tensor) 
{
    int nl = targets.size(0);
    int ne = targets.size(1);
 
    if (nl == 0) 
    {
        return torch::zeros({ batch_size, 0, ne - 1 },
            torch::TensorOptions().device(device));
    }

    auto i = targets.index({ torch::indexing::Slice(), 0 });
    // 原python代码： _, counts = i.unique(return_counts=True)
    auto [i_unique2_sort, i_unique2_inverse, counts] = torch::_unique2(i, true, false, true);
    counts = counts.to(torch::kInt32);
    auto out = torch::zeros({ batch_size, counts.max().item().toInt(), ne - 1 },
        torch::TensorOptions().device(device));
    // std::cout << "preprocess out " << out.sizes() << " counts: " << counts.sizes() <<
    //     " c_max: " << counts.max() << std::endl;
    for (int j = 0; j < batch_size; j++) {
        torch::Tensor matches = (i == j);
        int n = matches.sum().item<int>();
        if (n > 0) {
            out.index_put_({ j, torch::indexing::Slice(0, n) },
                targets.index({ matches, torch::indexing::Slice(1, torch::indexing::None) }));
        }
    }

    auto out_scale = out.index({ "...", torch::indexing::Slice(1, 5) });
    out_scale.mul_(scale_tensor);
    auto out_scale_xyxy = xywh2xyxy(out_scale);
    out.index_put_({ "...", torch::indexing::Slice(1, 5) }, out_scale_xyxy);
    // std::cout << "out: " << out[0][0] << std::endl;  //已经验证输入数据一致
    // std::cout << "out: " << out[batch_size-1][0] << std::endl;
    return out;
}

torch::Tensor v8DetectionLossImpl::bbox_decode(torch::Tensor anchor_points, torch::Tensor pred_dist)
{
    if (use_dfl)
    {
        auto b = pred_dist.size(0);
        auto a = pred_dist.size(1);
        auto c = pred_dist.size(2);

        //  pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        this->proj = this->proj.to(pred_dist.dtype());
        pred_dist = pred_dist.view({ b, a, 4, c / 4 }).softmax(3).matmul(proj.to(pred_dist.dtype()));
    }

    return dist2bbox(pred_dist, anchor_points, false);
}

/*
save feats0: torch.Size([16, 144, 80, 80]) as preds0.pt
save feats1: torch.Size([16, 144, 40, 40]) as preds1.pt
save feats2: torch.Size([16, 144, 20, 20]) as preds2.pt
save targets torch.Size([164, 6])
*/
std::tuple<torch::Tensor, torch::Tensor> v8DetectionLossImpl::forward(std::vector<torch::Tensor>& preds,
    VariantData& batch)
{
    auto loss = torch::zeros({ 3 }).to(device); // box, cls, dfl

    std::vector<torch::Tensor> feats = preds;
    std::vector<torch::Tensor> views;
    int feats0_size0 = feats[0].size(0);

    for (int i = 0; i < feats.size(); i++)
    {
        views.push_back(feats[i].view({ feats0_size0, this->no, -1 }));
    }

    auto cat_result = torch::cat(views, 2);
    auto cat_result_split = cat_result.split({ this->reg_max * 4, this->nc }, 1);
    torch::Tensor pred_distri = cat_result_split[0];
    torch::Tensor pred_scores = cat_result_split[1];

    pred_scores = pred_scores.permute({ 0, 2, 1 }).contiguous();    // [bs, 80, 8400] ==> [bs, 8400, 80]
    pred_distri = pred_distri.permute({ 0, 2, 1 }).contiguous();

    auto dtype = pred_scores.dtype();
    auto batch_size = pred_scores.size(0);
    auto feats0_shape = feats[0].sizes();
    torch::Tensor imgsz = torch::tensor(feats0_shape.slice(2),
        torch::TensorOptions().dtype(dtype).device(device));
    imgsz = imgsz * stride[0];

    // std::cout << "dtype: " << dtype << " batch_size: " << batch_size << 
    //     " imgsz: " << imgsz << " " << imgsz.dtype() << std::endl;

    auto [anchor_points, stride_tensor] = make_anchors(feats, this->stride, 0.5f);

    // std::cout << "anchor_points: " << anchor_points.sizes() << " stride_tensor " << stride_tensor.sizes() << std::endl;
    // std::cout << "anchor_points: " << anchor_points[0] << " " << anchor_points[1000] << std::endl;
    // std::cout << "stride_tensor: " << stride_tensor[0] << " " << stride_tensor[1000] << std::endl;

    // Targets
    torch::Tensor batch_idx = std::get<torch::Tensor>(batch[GET_VDK(VDK::BATCH_IDX)]);
    torch::Tensor cls = std::get<torch::Tensor>(batch[GET_VDK(VDK::CLS)]);
    torch::Tensor bboxes = std::get<torch::Tensor>(batch[GET_VDK(VDK::BBOXES)]);

    auto targets = torch::cat({ batch_idx.view({-1, 1}),
                                cls.view({-1, 1}),
                                bboxes }, 1);
    /*
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor = imgsz [[1, 0, 1, 0]] )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim = True).gt_(0.0)
    */

    auto scale_tensor = imgsz.index({torch::tensor({ 1, 0, 1, 0 })});
    //targets = targets.to(this->device, true);
    targets = preprocess(targets, batch_size, scale_tensor);    // [bs, -1, 6] ==> [bs, -1, 5]
    auto targets_split = targets.split({ 1, 4 }, 2);
    torch::Tensor gt_labels = targets_split[0];
    torch::Tensor gt_bboxes = targets_split[1];
    torch::Tensor mask_gt = gt_bboxes.sum(2, true).gt_(0.f);

    // Pboxes
    auto pred_bboxes = bbox_decode(anchor_points, pred_distri);
    //std::cout << "3 -- bbox_decode over... ret pred_bboxes: " << pred_bboxes.sizes() << std::endl;
    auto [target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx] = assigner->forward(
                                                                pred_scores.detach().sigmoid(),
                                                                (pred_bboxes.detach() * stride_tensor).to(gt_bboxes.dtype()),
                                                                anchor_points * stride_tensor,
                                                                gt_labels,
                                                                gt_bboxes,
                                                                mask_gt);
    //std::cout << "4 -- assigner over...\n";
#if 0    
    if(std::get<bool>(hyp->at("debug_loss_enable")))
    {
        pred_distri = load_tensordata_from_file("pred_distri.pt");
        pred_bboxes = load_tensordata_from_file("pred_bboxes.pt");
        anchor_points = load_tensordata_from_file("anchor_points.pt");
        target_bboxes = load_tensordata_from_file("target_bboxes.pt");
        target_scores = load_tensordata_from_file("target_scores.pt");
        fg_mask = load_tensordata_from_file("fg_mask.pt");

        std::cout << "load pred_distri... form file." << std::endl;
    }
#endif
    // 正负样本都参与了损失计算
    float target_scores_sum = std::max(target_scores.sum().item<float>(), 1.0f);

    loss[1] = this->bce->forward(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum;
    //std::cout << "cls loss: " << loss[1] << std::endl;
    //std::cout << "5 -- bce over...\n";    
    // std::cout << "fg_mask: " << fg_mask.sizes() << " " << fg_mask.sum() << std::endl;    
    if (fg_mask.sum().item().toInt())
    {
        // std::cout << "pred_distri: " << pred_distri.sizes() << std::endl;
        // std::cout << "pred_bboxes: " << pred_bboxes.sizes() << std::endl;
        // std::cout << "anchor_points: " << anchor_points.sizes() << std::endl;
        // std::cout << "target_bboxes: " << target_bboxes.sizes() << std::endl;
        // std::cout << "target_scores: " << target_scores.sizes() << std::endl;
        // std::cout << "target_scores_sum: " << target_scores_sum << std::endl;

        target_bboxes /= stride_tensor;
        auto [tmp_lbox, tmp_ldfl] = bbox_loss->forward(
            pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
            target_scores_sum, fg_mask);

        loss[0] = tmp_lbox;
        loss[2] = tmp_ldfl;
    }
    //std::cout << "6 -- box loss over...\n";   
  
    loss[0] = loss[0] * box_ratio;
    loss[1] = loss[1] * cls_ratio;
    loss[2] = loss[2] * dfl_ratio;
    return { loss.sum() * batch_size, loss.detach() };
}

void v8DetectionLossImpl::createLossRecord()
{}

std::string v8DetectionLossImpl::getLossitemsNames()
{
    char temp_s[120];
    sprintf(temp_s, "%9s%7s%10s%10s%10s%10s %7s",
        "epochs", "idx", "loss", "boxs", "cls", "dfl", "target");
    return std::string(temp_s);
}

std::string v8DetectionLossImpl::getLossitemsInfo(int epoch, int epochs, int idx, int num_targets)
{
    auto lossitems = m_lossitems / std::max(1.0f, float(m_times));
    char temp_s[120];
    sprintf(temp_s, "%4d-%-4d%7d%10.5f%10.5f%10.5f%10.5f %6d",
        epoch+1, epochs, idx,
        m_loss.item<float>(),
        lossitems[0].item().toFloat(),
        lossitems[1].item().toFloat(),
        lossitems[2].item().toFloat(),
        num_targets
    );

    return std::string(temp_s);
}


//--------------------------- end V8DetectionLoss --------------------------

