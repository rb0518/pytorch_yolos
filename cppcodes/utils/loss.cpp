#include "loss.h"
#include "general.h"
#include "utils.h"

//  c10::Dict<std::string, torch::IValue> hyp;
#include <torch/torch.h>
#include <cmath>

//--------------------------- start V8DetectionLoss --------------------------
v8DetectionLossImpl::v8DetectionLossImpl(std::shared_ptr<ModelImpl> model, int tal_topk /*= 10*/)
{
    this->device = torch::Device(torch::kCUDA);

    bce = torch::nn::BCEWithLogitsLoss(
        torch::nn::BCEWithLogitsLossOptions().reduction(torch::kNone));
    std::shared_ptr<DetectImpl> m = model->last_module;

    hyp = model->hyp;
    if(model->hyp.size())
        for (auto& [k, v] : model->hyp)
        {
            hyp[k] = v;
        }

    //show_cfg_info("v8 hyp", hyp);

    this->stride = m->stride;
    this->nc = m->nc;
    this->no = m->nc + m->reg_max * 4;
    this->reg_max = m->reg_max;

    this->use_dfl = m->reg_max > 1;
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

std::tuple<torch::Tensor, torch::Tensor> v8DetectionLossImpl::forward(std::vector<torch::Tensor>& preds,
    torch::Dict<std::string, torch::Tensor>& batch)
{
    auto loss = torch::zeros({ 3 }).to(device); // box, cls, dfl
    /*
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
    */
    std::vector<torch::Tensor> feats = preds;
    std::vector<torch::Tensor> views;
    int feats0_size0 = feats[0].size(0);
    for (int i = 0; i < feats.size(); i++)
    {
        auto xi = feats[i].clone();
        views.push_back(xi.view({ feats0_size0, this->no, -1 }));
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
    torch::Tensor img_size = torch::tensor(feats0_shape.slice(2),
        torch::TensorOptions().dtype(dtype).device(device));
    img_size = img_size * stride[0];
    auto [anchor_points, stride_tensor] = make_anchors(feats, this->stride, 0.5f);

    // Targets
    auto targets = torch::cat({ batch.at("batch_idx").view({-1, 1}),
                                batch.at("cls").view({-1, 1}),
                                batch.at("bboxes") }, 1);
    /*
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor = imgsz [[1, 0, 1, 0]] )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim = True).gt_(0.0)
    */
    //std::cout << "img_size: " << img_size << std::endl;
    auto scale_tensor = img_size.index({torch::tensor({ 1, 0, 1, 0 })});
    targets = targets.to(this->device);
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
    // std::cout << "4 -- assigner over...\n";

    // 正负样本都参与了损失计算
    float target_scores_sum = std::max(target_scores.sum().item<float>(), 1.0f);
    loss[1] = this->bce->forward(pred_scores, target_scores.to(dtype)).sum().item<float>() / target_scores_sum;

    //std::cout << "5 -- bce over...\n";        
    if (fg_mask.sum().item().toInt())
    {
        target_bboxes /= stride_tensor;
        auto [tmp_lbox, tmp_ldfl] = bbox_loss->forward(
            pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
            target_scores_sum, fg_mask);

        loss[0] = tmp_lbox;
        loss[2] = tmp_ldfl;
    }

    hyp["dfl"] = 1.5f;
    hyp["box"] = 7.5f;
    hyp["cls"] = 0.5f;
    loss[0] = loss[0].item().toFloat() * std::get<float>(hyp["box"]);
    loss[1] = loss[1].item().toFloat() * std::get<float>(hyp["cls"]);
    loss[2] = loss[2].item().toFloat() * std::get<float>(hyp["dfl"]);
    return { loss.sum() * batch_size, loss.detach() };
}
//--------------------------- end V8DetectionLoss --------------------------

