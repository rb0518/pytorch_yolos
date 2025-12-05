#include <torch/torch.h>
#include <vector>

std::vector<torch::Tensor> non_max_suppression(
    torch::Tensor prediction,
    float conf_thres = 0.25,
    float iou_thres = 0.45,
    const std::vector<std::vector<int>>& classes = {},
    bool agnostic = false,
    bool multi_label = false,
    const std::vector<std::vector<float>>& labels = {},
    int max_det = 300,
    int nm = 0  // number of masks
);

namespace ops {
    /*
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    */
    std::vector<torch::Tensor> non_max_suppression(
        torch::Tensor prediction,
        float conf_thres = 0.25f,
        float iou_thres = 0.45f,
        std::vector<int> classes = {},
        bool agnostic = false,
        bool multi_label = false,
        std::vector<int> = {}, // 暂时未用，用std::vector<int>代替
        int max_det = 300,
        int nc = 0,
        float max_time_img = 0.05f,
        int max_nms = 30000,
        int max_wh = 7680,
        bool in_place = true,
        bool rotated = false
        );
} //end namespace ops