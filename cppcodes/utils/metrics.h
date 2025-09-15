#pragma once
#include <torch/torch.h>
#include <vector>

torch::Tensor fitness(torch::Tensor x);

std::tuple<double, torch::Tensor, torch::Tensor> compute_ap(
    const torch::Tensor& recall,
    const torch::Tensor& precision,
    const std::string& method = "interp");

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ap_per_class(
    const torch::Tensor& tp,
    const torch::Tensor& conf,
    const torch::Tensor& pred_cls,
    const torch::Tensor& target_cls,
    bool plot = false,
    const std::string& save_dir = ".",
    const std::vector<std::string>& names = {});

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