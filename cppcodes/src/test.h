#pragma once

#include <torch/torch.h>

#include "yolo.h"
#include "torch_utils.h"
/*
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False):
*/

#include "datasets.h"

Dataloader_Custom test(
    std::shared_ptr<ModelImpl> model,   // model or nullpytr
    std::string root_,          
    VariantConfigs opt,             // opt 
    std::vector<std::string> cls_names,
    Dataloader_Custom val_dataloader, // if not val, set = nullptr
    int val_total_number,
    std::string val_path,
    std::string save_dir_,
    std::string weights = "",
    int nc = 80,
    int imgsz = 640,
    int batch_size=32,
    float conf_thres = 0.001f,
    float iou_thres = 0.6f,
    bool save_pred = false
);

class BaseValidator
{
public:
    BaseValidator(std::shared_ptr<Model> _model_ptr,
        VariantConfigs _args_input, std::string _root_dir);

    void setup_model();

    void init_device();
    void init_dirs();

    void load_weight();

    void init_dataloader();

    void do_validate();
protected:
    YoloCustomExample preprocess(YoloCustomExample& batch);
    std::vector<torch::Dict<std::string, torch::Tensor>> postprocess(torch::Tensor preds);

    void update_metrics(std::vector<torch::Dict<std::string, torch::Tensor>> preds,
        torch::Dict<std::string, torch::Tensor> batch);

    void _model_eval();

    torch::Dict<std::string, torch::Tensor> _prepare_batch(int si,
        torch::Dict<std::string, torch::Tensor> batch);

    torch::Dict<std::string, torch::Tensor> _prepare_pred(
        torch::Dict<std::string, torch::Tensor> pred);
public:
    std::shared_ptr<Model> model_ptr = { nullptr };
    bool is_training = false;

    std::string root_dir;
    VariantConfigs args;
    std::string task_name;

    torch::Device device = torch::Device(torch::kCPU);

    std::string save_dir;
    int nc = 80;
    int imgsz = 640;

    DataloaderBase* val_loader;

    int seen;
};