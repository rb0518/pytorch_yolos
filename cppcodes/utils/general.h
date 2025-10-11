#pragma once
#include <string>
#include "yaml_load.h"
// 取得根目录路径
std::string get_root_path_string();
// 从cfgs目录中读取默认环境配置
void load_default_environment(const std::string& root_path, VariantConfigs& opts);

std::string increment_path(const std::string prj_and_name, bool exist_ok = true, std::string sep = "");

// std::string get_last_run(std::string search_dir)
std::string get_last_run(std::string search_dir);

torch::Tensor bbox_iou(torch::Tensor box1_, torch::Tensor box2_, bool is_xywh= true,
    bool GIoU = false, bool DIoU = false, bool CIoU = false, float eps = 1e-7);

torch::Tensor box_iou(const torch::Tensor & boxes1, const torch::Tensor & boxes2);    

torch::Tensor segment2box(torch::Tensor& segments, int width, int height);
std::vector<torch::Tensor> resample_segments(std::vector<torch::Tensor>& segments, int n = 1000);


torch::Tensor process_mask(const torch::Tensor& protos,           
                            const torch::Tensor& masks_in,         
                            const torch::Tensor& bboxes,           
                            const std::vector<int64_t>& shape,  
                            bool upsample = false);          

torch::Tensor crop_mask(torch::Tensor masks, torch::Tensor boxes);                            