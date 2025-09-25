#pragma once

#include <torch/torch.h>
#include <vector>

#include "Yolo.h"
#include "yaml_load.h"
/*
class FocalLossImpl : public torch::nn::Module {
public:
    FocalLossImpl(torch::nn::BCEWithLogitsLoss& loss_fcn,
                 double gamma = 1.5, 
                 double alpha = 0.25)
        : gamma_(gamma),
          alpha_(alpha) 
    {
        original_options_ = loss_fcn->options;
        auto new_options = torch::nn::BCEWithLogitsLossOptions().weight(original_options_.weight())
            .pos_weight(original_options_.pos_weight())
            .reduction(torch::kNone);
        auto device_ = loss_fcn->parameters().begin()->device();
        loss_fcn_ = torch::nn::BCEWithLogitsLoss(new_options, device_);
    }

    torch::Tensor forward(torch::Tensor pred, torch::Tensor true_tensor) {
        auto loss = loss_fcn_->forward(pred, true_tensor);
        
        // TensorFlow implementation style
        auto pred_prob = torch::sigmoid(pred);  // prob from logits
        auto p_t = true_tensor * pred_prob + (1 - true_tensor) * (1 - pred_prob);
        auto alpha_factor = true_tensor * alpha_;
        
        // Calculate focal loss
        auto modulating_factor = torch::pow(1.0 - p_t, gamma_);
        loss = loss * alpha_factor * modulating_factor;
        
        // Apply original reduction
        auto reduction_type = original_options_.reduction();
        if (reduction_type.index() == torch::Reduction::Mean)
        {
            return loss.mean();
        } 
        else if (reduction_type.index() == torch::Reduction::Sum)
        {
            return loss.sum();
        }
        return loss;  // none reduction
    }

private:
    torch::nn::BCEWithLogitsLoss loss_fcn_{nullptr};
    double gamma_;
    double alpha_;
    torch::nn::BCEWithLogitsLossOptions original_options_;
};
TORCH_MODULE(FocalLoss);
*/
class ComputeLoss// : public torch::nn::Module
{
public:
    bool sort_obj_iou = false;

    ComputeLoss(Detect m, VariantConfigs& _hyp, bool autobalance = false);

    std::tuple<torch::Tensor, torch::Tensor> operator()(const std::vector<torch::Tensor>& p,
        const torch::Tensor& targets);

private:
    torch::nn::BCEWithLogitsLoss BCEcls{nullptr}, BCEobj{nullptr};
    ///FocalLoss FocalLoss_cls{nullptr}, FocalLoss_obj{nullptr};
    double cp, cn, gr;

    std::vector<double> balance;
    int ssi, na, nc, nl;
    //torch::Tensor anchors;
    torch::Device _device = torch::Device(torch::kCPU);
    bool autobalance;
    torch::Tensor strides;

    std::tuple<std::vector<torch::Tensor>,
        std::vector<torch::Tensor>,
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>,
        std::vector<torch::Tensor>>
        build_targets(const std::vector<torch::Tensor>& p, 
            const torch::Tensor& _targets); 

    Detect m_ptr{nullptr};
    VariantConfigs hyp;
};

