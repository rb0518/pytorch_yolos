#pragma once
#include <torch/torch.h>
#include <vector>

std::tuple<double, torch::Tensor, torch::Tensor> compute_ap(
    const torch::Tensor& recall,
    const torch::Tensor& precision,
    const std::string& method = "interp");

#if 0
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ap_per_class(
    const torch::Tensor& tp,
    const torch::Tensor& conf,
    const torch::Tensor& pred_cls,
    const torch::Tensor& target_cls,
    bool plot = false,
    const std::string& save_dir = ".",
    const std::vector<std::string>& names = {});
#else
std::tuple<
    torch::Tensor,  // tp
    torch::Tensor,  // fp
    torch::Tensor,  // p
    torch::Tensor,  // r
    torch::Tensor,  // f1

    torch::Tensor,  // ap
    torch::Tensor,  // unique_classes
    torch::Tensor,  // p_curve
    torch::Tensor,  // r_curve
    torch::Tensor,  // f1_curve

    torch::Tensor,  // x
    torch::Tensor   // prec_values
>
ap_per_class(
    const torch::Tensor& tp_,
    const torch::Tensor& conf_,
    const torch::Tensor& pred_cls_,
    const torch::Tensor& target_cls_,
    bool plot = false,
    const std::string& save_dir = ".",
    const std::vector<std::string>& names = {},
    float eps = 1e-16);
#endif


torch::Tensor process_batch(torch::Tensor& detections, torch::Tensor& labels, torch::Tensor& iouv);
/*
class ConfusionMatrix 
{
public:
    ConfusionMatrix(int nc, float conf = 0.25f, float iou_thres = 0.45f);

    void process_batch(const torch::Tensor& detections, const torch::Tensor& labels);

    torch::Tensor get_matrix();

    void print();

    void plot(const std::string& save_dir = ".", const std::vector<std::string>& names = {});

private:
    torch::Tensor matrix_; // (nc+1) x (nc+1) 
    int nc_; // num_classes
    float conf_thres_; 
    float iou_thres_; 
};
*/