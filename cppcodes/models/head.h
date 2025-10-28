#pragma once

/*
    # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
    __all__ =   
                "Detect",           [√]
                "Segment", 
                "Pose", 
                "Classify", 
                "OBB", 
                "RTDETRDecoder", 
                "v10Detect", 
                "YOLOEDetect", 
                "YOLOESegment"
                
*/

#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <unordered_map>

#include "head.h"
#include "block.h"

/*
    """
    YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.
*/
// 完全修改完成后，再修改回类名
class DetectImpl : public torch::nn::Module 
{
public:
    DetectImpl(int _nc, std::vector<int> _ch);  

    std::tuple<torch::Tensor, std::vector<torch::Tensor>>
        forward(std::vector<torch::Tensor> x);

    std::tuple<torch::Tensor, std::unordered_map<std::string, std::vector<torch::Tensor>>>
        forward_end2end(std::vector<torch::Tensor>& x);

    at::Tensor _inference(std::vector<at::Tensor> x);
    at::Tensor decode_bboxes(torch::Tensor bboxes, torch::Tensor anchors, bool xywh = true);
    void bias_init();

    torch::Tensor postprocess(torch::Tensor preds, int max_det, int nc = 80); 

public:
    int nc;             // number of classes
    int nl;             // number of detection layers
    int reg_max;        // DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
    int no;             // number of ouputs per anchors
    at::Tensor stride;  // strides computed during build

    int c2;
    int c3;

    torch::nn::ModuleList cv2;  // layers for box regression
    torch::nn::ModuleList cv3;  // layers for classification

    torch::nn::AnyModule dfl;   // Distribution Focal Loss layer

    torch::nn::ModuleList one2one_cv2;  // One-to-one convolution layers for box regression
    torch::nn::ModuleList one2one_cv3;  // One-to-one convolution layers for classification

    // default sets;
    bool _dynamic_ = false;     // force grid reconstruction
    bool _export_ = false;      // export mode
    bool end2end = false;       // end2end
    
    int max_det = 300;          // max_det
    c10::IntArrayRef shape_;
    at::Tensor anchors = torch::empty({ 0 });
    at::Tensor strides = torch::empty({ 0 });
    bool legacy = false;        // backward compatiblibity for v3/v5/v8/v9 models
    bool xyxy = false;          // xyxy or xywh ouput
};
TORCH_MODULE(Detect);