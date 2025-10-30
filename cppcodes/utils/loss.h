#pragma once

#include <torch/torch.h>
#include <vector>

#include "Yolo.h"
#include "yaml_load.h"
/*
// 2025-10-17 Add V8DetectionLoss 
    """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
*/
#include "head.h"
#include "tal.h"
/*
    """
    Return sum of left and right DFL losses.

    Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
    https://ieeexplore.ieee.org/document/9792391
    """
*/
class DFLossImpl : public torch::nn::Module
{
public:
    DFLossImpl(int _reg_max) : reg_max(_reg_max)
    {
    }

    torch::Tensor forward(torch::Tensor pred_dist, torch::Tensor target)
    {
        target = target.clamp_(0, reg_max - 1.01f);
        auto tl = target.to(torch::kLong);
        auto tr = tl + 1;
        auto wl = tr - target;
        auto wr = 1 - wl;
        
        auto ce_1 = torch::nn::functional::cross_entropy(pred_dist, tl.view(-1),
            torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone)).view(tl.sizes()) * wl;

        auto ce_2 = torch::nn::functional::cross_entropy(pred_dist, tr.view(-1),
            torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone)).view(tl.sizes()) * wr;
        return (ce_1 + ce_2).mean(-1, true);
    }

public:
    int reg_max;
};
TORCH_MODULE(DFLoss);

class BboxLossImpl : public torch::nn::Module
{
public:
    BboxLossImpl(int reg_max) {
        if (reg_max > 1)
            dfl_loss = DFLoss(reg_max);
    }

    std::tuple<torch::Tensor, torch::Tensor>
        forward(
            torch::Tensor pred_dist,
            torch::Tensor pred_bboxes,
            torch::Tensor anchor_points,
            torch::Tensor target_bboxes,
            torch::Tensor target_scores,
            float target_scores_sum,
            torch::Tensor fg_mask)
    { 
        auto weight = target_scores.sum(-1).index({ fg_mask }).unsqueeze(-1);
        //   iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        auto iou = bbox_iou(pred_bboxes.index({ fg_mask }), target_bboxes.index({ fg_mask }), false, false, false, true);
        auto loss_iou = ((1.0f - iou) * weight).sum() / target_scores_sum;
        /*
                # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
*/
        torch::Tensor loss_dfl = torch::tensor(0.0).to(pred_dist.device());
        if (dfl_loss)
        {
            auto target_ltrb = bbox2dist(anchor_points, target_bboxes, dfl_loss->reg_max - 1);
            loss_dfl = dfl_loss->forward(pred_dist.index({ fg_mask }).view({ -1, dfl_loss->reg_max }),
                target_ltrb.index({ fg_mask }));
            loss_dfl *= weight;
            loss_dfl = loss_dfl.sum() / target_scores_sum;
        }

        return { loss_iou, loss_dfl };
    }
public:
    DFLoss dfl_loss{ nullptr };
};
TORCH_MODULE(BboxLoss);

class v8DetectionLossImpl
{
public:
    v8DetectionLossImpl(std::shared_ptr<ModelImpl> model, int tal_topk = 10);
    std::tuple<torch::Tensor, torch::Tensor> forward(std::vector<torch::Tensor>& preds,
         torch::Dict<std::string, torch::Tensor>& batch);
public:
    torch::Device device = torch::Device(torch::kCPU);
    torch::nn::BCEWithLogitsLoss bce{ nullptr };
    torch::Tensor stride;
    int nc;
    int no;
    int reg_max;
    bool use_dfl;
    VariantConfigs* hyp;

    TaskAlignedAssigner assigner{ nullptr };
    BboxLoss bbox_loss{ nullptr };
    torch::Tensor proj;

private:
    torch::Tensor preprocess(torch::Tensor targets, int batch_size, torch::Tensor scale_tensor);
    torch::Tensor bbox_decode(torch::Tensor anchor_points, torch::Tensor pred_dist);

    bool save_tensor_firsttime = false;
};
TORCH_MODULE(v8DetectionLoss);



