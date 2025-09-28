#pragma once

#include <torch/torch.h>

#include "yolo.h"
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

void test(
    std::shared_ptr<Model> model,   // model or nullpytr
    std::string root_,          
    VariantConfigs opt,             // opt 
    std::vector<std::string> cls_names,
    Dataloader_Detect val_dataloader, // if not val, set = nullptr
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