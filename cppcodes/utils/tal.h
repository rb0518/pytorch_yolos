#pragma once

#include <torch/torch.h>
#include "general.h"

std::tuple<torch::Tensor, torch::Tensor>
make_anchors(const std::vector<torch::Tensor>& feats,
    const torch::Tensor& strides,
    float grid_cell_offset = 0.5f);

//  """Transform distance(ltrb) to box(xywh or xyxy)."""
torch::Tensor dist2bbox(torch::Tensor distance, torch::Tensor anchor_points, bool xywh = true, int dim = -1);

//   """Transform bbox(xyxy) to dist(ltrb)."""
torch::Tensor bbox2dist(torch::Tensor anchor_points, torch::Tensor bbox, int reg_max);

torch::Tensor dist2rbox(torch::Tensor pred_dist, torch::Tensor pred_angle, torch::Tensor anchor_points, int dim = -1);

/*
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """
*/
class TaskAlignedAssignerImpl : public torch::nn::Module
{
public:
    explicit TaskAlignedAssignerImpl(int _topk = 13, int _num_classes = 80, 
        float _alpha = 1.0f, float _beta = 6.0, float _eps = 1e-9);

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
        forward(
            torch::Tensor pd_scores,
            torch::Tensor pd_bboxes,
            torch::Tensor anc_points,
            torch::Tensor gt_labels,
            torch::Tensor gt_bboxes,
            torch::Tensor mask_gt);
public:
    int topk;
    int num_classes;
    float alpha;
    float beta;
    float eps;

    int64_t bs;
    int64_t n_max_boxes;
private:
    std::tuple<at::Tensor, at::Tensor, at::Tensor> get_pos_mask(
        torch::Tensor pd_scores,
        torch::Tensor pd_bboxes,
        torch::Tensor gt_labels,
        torch::Tensor gt_bboxes,
        torch::Tensor anc_points,
        torch::Tensor mask_gt);

    std::tuple<torch::Tensor, torch::Tensor>
        get_box_metrics(torch::Tensor pd_scores, torch::Tensor pd_bboxes, torch::Tensor gt_labels,
            torch::Tensor gt_bboxes, torch::Tensor mask_gt);

    torch::Tensor iou_calculation(torch::Tensor gt_bboxes,
        torch::Tensor pd_bboxes);

    torch::Tensor select_candidates_in_gts(torch::Tensor xy_centers,
        torch::Tensor gt_bboxes, float eps = 1e-9);

    torch::Tensor select_topk_candidates(torch::Tensor metrics, bool largest = true,
        c10::optional<torch::Tensor> topk_mask = c10::nullopt);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        select_highest_overlaps(torch::Tensor mask_pos,
            torch::Tensor overlaps,
            int n_max_boxes);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        get_target(torch::Tensor gt_labels,
            torch::Tensor gt_bboxes,
            torch::Tensor target_gt_idx,
            torch::Tensor fg_mask);
};
TORCH_MODULE(TaskAlignedAssigner);

