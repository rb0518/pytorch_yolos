#include "tal.h"
#include "general.h"        

std::tuple<torch::Tensor, torch::Tensor> get_2_splits(torch::Tensor x, int split_size, int dim)
{
    auto s = x.split(split_size, dim);
    if (s.size() != 2) LOG(ERROR) << "split_size error, return not eq 2";
    return std::make_tuple(s[0], s[1]);
}

std::tuple<torch::Tensor, torch::Tensor>
make_anchors(const std::vector<torch::Tensor>& feats,
    const torch::Tensor& strides,
    float grid_cell_offset/* = 0.5f*/)
{
    std::vector<torch::Tensor> anchor_points;
    std::vector<torch::Tensor> stride_tensors;

    auto dtype = feats[0].dtype();
    auto device = feats[0].device();
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    for (size_t i = 0; i < strides.size(0); i++)
    {
        auto stride = strides[i].item().toInt();
        auto feat = feats[i];
        //std::cout << i << " feat: " << feat.sizes() << " stride " << stride << std::endl;
        auto h = feat.size(2);
        auto w = feat.size(3);
        auto sx = torch::arange(w, options) + grid_cell_offset;
        auto sy = torch::arange(h, options) + grid_cell_offset;
        
        auto yvxv = torch::meshgrid({ sy, sx }, "ij");
        sy = yvxv[0];
        sx = yvxv[1];
        anchor_points.push_back(torch::stack({ sx, sy }, -1).view({ -1, 2 }));
        stride_tensors.push_back(torch::full({ h * w, 1 }, stride, options));
    }
    return { torch::cat(anchor_points), torch::cat(stride_tensors) };
}

torch::Tensor dist2bbox(torch::Tensor distance, torch::Tensor anchor_points, bool xywh /*= true*/, int dim /*= -1*/)
{
    auto distance_chunk = distance.chunk(2, dim);
    auto lt = distance_chunk[0];
    auto rb = distance_chunk[1];
    auto x1y1 = anchor_points - lt;
    auto x2y2 = anchor_points + rb;
    if (xywh)
    {
        auto c_xy = (x1y1 + x2y2) / 2;
        auto wh = x2y2 - x1y1;
        return torch::cat({ c_xy, wh }, dim);   // return xywh bbox
    }

    return torch::cat({ x1y1, x2y2 }, dim);     // return xyxy bbox
}

/*
def bbox2dist(anchor_points, bbox, reg_max) :
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist(lt, rb)
*/
torch::Tensor bbox2dist(torch::Tensor anchor_points, torch::Tensor bbox, int reg_max)
{
    auto bbox_chunk = bbox.chunk(2, -1);
    auto x1y1 = bbox_chunk[0];
    auto x2y2 = bbox_chunk[1];
    return torch::cat({ anchor_points - x1y1, x2y2 - anchor_points }, -1).clamp_(0, reg_max - 0.01f);
}
/*
*   """
    Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance with shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle with shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        dim (int, optional): Dimension along which to split.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes with shape (bs, h*w, 4).
    """
*/
torch::Tensor dist2rbox(torch::Tensor pred_dist, torch::Tensor pred_angle, torch::Tensor anchor_points, int dim /*= -1*/)
{
    auto [lt, rb] = get_2_splits(pred_dist, 2, dim);

    auto t_cos = torch::cos(pred_angle);
    auto t_sin = torch::sin(pred_angle);
    // # (bs, h*w, 1)
    auto [xf, yf] = get_2_splits((rb - lt) / 2, 1, dim);
    auto x = xf * t_cos - yf * t_sin;
    auto y = xf * t_sin + yf * t_cos;
    auto xy = torch::cat({ x, y }, dim) + anchor_points;
    return torch::cat({ xy, lt + rb }, dim);
}


TaskAlignedAssignerImpl::TaskAlignedAssignerImpl(int _topk, int _num_classes, float _alpha, float _beta, float _eps)
	: topk(_topk), num_classes(_num_classes), alpha(_alpha), beta(_beta), 
	eps(_eps)
{
}

/*
    """
    Compute the task-aligned assignment. Reference code is available at
    https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

    Args:
        pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
        pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
        anc_points (Tensor): shape(num_total_anchors, 2)
        gt_labels (Tensor): shape(bs, n_max_boxes, 1)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
        mask_gt (Tensor): shape(bs, n_max_boxes, 1)

    Returns:
        target_labels (Tensor): shape(bs, num_total_anchors)
        target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
        target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
        fg_mask (Tensor): shape(bs, num_total_anchors)
        target_gt_idx (Tensor): shape(bs, num_total_anchors)
    """
*/
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
TaskAlignedAssignerImpl::forward(torch::Tensor pd_scores,
    torch::Tensor pd_bboxes,
    torch::Tensor anc_points,
    torch::Tensor gt_labels,
    torch::Tensor gt_bboxes,
    torch::Tensor mask_gt) 
{
    torch::NoGradGuard nograd;

    // std::cout << "TaskAlignedAssignerImpl::forward input pd_scores: " <<
    //     pd_scores.sizes() << " pd_bboxes: " << pd_bboxes.sizes()
    //     << " anc_points: " << anc_points.sizes() << " gt_labels " << gt_labels.sizes()
    //     << " gt_bboxes " << gt_bboxes.sizes() << " mask_gt " << mask_gt.sizes() << std::endl;

    this->bs = pd_scores.size(0);
    this->n_max_boxes = gt_bboxes.size(1);

    if (this->n_max_boxes == 0)
    {
        auto _dev = gt_bboxes.device();
        return std::make_tuple(
            torch::full_like(pd_scores.index({"...", 0}), this->num_classes).to(_dev),
            torch::zeros_like(pd_bboxes).to(_dev),
            torch::zeros_like(pd_scores).to(_dev),
            torch::zeros_like(pd_scores.index({"...", 0})).to(_dev),
            torch::zeros_like(pd_scores.index({ "...", 0 })).to(_dev)
            );
    }

    auto [mask_pos, align_metric, overlaps] = get_pos_mask(pd_scores, pd_bboxes, 
                        gt_labels, gt_bboxes, anc_points, mask_gt);
    //std::cout << "1-- get_pos_mask over...\n";

    torch::Tensor target_gt_idx, fg_mask;
    std::tie(target_gt_idx, fg_mask, mask_pos) = select_highest_overlaps(
                mask_pos, overlaps, this->n_max_boxes);
    //std::cout << "2-- select_highest_overlaps \n";

    auto [target_labels, target_bboxes, target_scores] = get_target(
                gt_labels, gt_bboxes, target_gt_idx, fg_mask);
    //std::cout << "3-- get_target \n";
    align_metric *= mask_pos;
    auto pos_align_metrics = align_metric.amax(-1, true);
    auto pos_overlaps = (overlaps * mask_pos).amax(-1, true);
    torch::Tensor norm_algin_metrics = (align_metric * pos_overlaps / (pos_align_metrics + this->eps)).amax(-2).unsqueeze(-1);
    target_scores = target_scores *  norm_algin_metrics;
    //std::cout << "4-- return at here \n";
    
    return { target_labels, target_bboxes, target_scores, fg_mask.to(torch::kBool), target_gt_idx };
}

/*
    """Get in_gts mask, (b, max_num_obj, h*w)."""
*/
std::tuple<at::Tensor, at::Tensor, at::Tensor> 
TaskAlignedAssignerImpl::get_pos_mask(torch::Tensor pd_scores, 
    torch::Tensor pd_bboxes, 
    torch::Tensor gt_labels, 
    torch::Tensor gt_bboxes, 
    torch::Tensor anc_points, 
    torch::Tensor mask_gt)
{
    auto mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes);
    //std::cout << " get_pos_mask 1-- over: " << mask_in_gts.sizes() << std::endl;
    auto [align_metric, overlaps] = get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt);
    //std::cout << " get_pos_mask 2 -- over" << align_metric.sizes() << " " << overlaps.sizes() << std::endl;
    auto topk_mask = mask_gt.expand({ -1, -1, this->topk }).to(torch::kBool);
    auto mask_topk = select_topk_candidates(align_metric, true, topk_mask);
    //std::cout << " get_pos_mask 3 -- over"<< std::endl;
    auto mask_pos = mask_topk * mask_in_gts * mask_gt;
    return {mask_pos, align_metric, overlaps};
}

std::tuple<torch::Tensor, torch::Tensor> 
TaskAlignedAssignerImpl::get_box_metrics(torch::Tensor pd_scores, 
    torch::Tensor pd_bboxes, 
    torch::Tensor gt_labels,
    torch::Tensor gt_bboxes, 
    torch::Tensor mask_gt)
{
    int na = pd_bboxes.size(-2);
    // std::cout << "pd_bboxes " << pd_bboxes.sizes() << " na " << std::endl;
    // std::cout << " pd_scores " << pd_scores.sizes() << std::endl;

    mask_gt = mask_gt.to(torch::kBool);
    
    torch::Tensor overlaps = torch::zeros({ this->bs, this->n_max_boxes, na },
        torch::TensorOptions().dtype(pd_bboxes.dtype()).device(pd_bboxes.device()));
    torch::Tensor bbox_scores = torch::zeros({this->bs, this->n_max_boxes, na}, 
        torch::TensorOptions().dtype(pd_scores.dtype()).device(pd_scores.device()));

    torch::Tensor ind = torch::zeros({ 2, this->bs, this->n_max_boxes }, 
        torch::TensorOptions().dtype(torch::kLong));
    auto ind_range = torch::arange(bs).view({ -1, 1 }).expand({ -1, this->n_max_boxes });
    //ind.index_put_({ 0 }, ind_range);
    ind[0] = ind_range;
    //ind.index_put_({ 1 }, gt_labels.squeeze(-1));
    ind[1] = gt_labels.squeeze(-1);
    //std::cout << "ind[0] " << ind[0].sizes() << " ind[1] " << ind[1].sizes() << std::endl;
    /*
        # Get the scores of each grid for each gt cls
        # b, max_num_obj, h * w
    */
    bbox_scores.index_put_({ mask_gt }, pd_scores.index({
        ind[0],
        torch::indexing::Slice(),
        ind[1] }).index({ mask_gt }));
    //std::cout << " get_box_metrics 2 over..." << std::endl;
    //         # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
    pd_bboxes = pd_bboxes.unsqueeze(1)
        .expand({ -1, this->n_max_boxes, -1, -1 }).index({ mask_gt });
    gt_bboxes = gt_bboxes.unsqueeze(2).expand({ -1, -1, na, -1 }).index({ mask_gt });
    overlaps.index_put_({ mask_gt }, iou_calculation(gt_bboxes, pd_bboxes));
    //std::cout << " get_box_metrics 3 over..." << std::endl;

    auto align_mask = bbox_scores.pow(this->alpha) * overlaps.pow(this->beta);
    return { align_mask, overlaps };
}

torch::Tensor TaskAlignedAssignerImpl::iou_calculation(torch::Tensor gt_bboxes, torch::Tensor pd_bboxes)
{
/*
//  return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
//    torch::Tensor bbox_iou(torch::Tensor box1_, torch::Tensor box2_, bool is_xywh = false,
//        bool GIoU = false, bool DIoU = false, bool CIoU = false, float eps = 1e-7)
*/
    return bbox_iou(gt_bboxes, pd_bboxes, false, false, false, true).squeeze(-1).clamp_(0);
}

/*
    """
    Select positive anchor centers within ground truth bounding boxes.

    Args:
        xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
        gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
        eps (float, optional): Small value for numerical stability. Defaults to 1e-9.

    Returns:
        (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

    Note:
        b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
        Bounding box format: [x_min, y_min, x_max, y_max].
    """
*/
torch::Tensor TaskAlignedAssignerImpl::select_candidates_in_gts(torch::Tensor xy_centers,
    torch::Tensor gt_bboxes, float eps/*=1e-9*/)
{
    //std::cout << " select_candidates_in_gts input: " << xy_centers.sizes() << " gt_bboxes: " << gt_bboxes.sizes() << std::endl;
    int n_anchors = xy_centers.size(0);
    int bs_ = gt_bboxes.size(0);
    int n_boxes = gt_bboxes.size(1);
    auto gt_bboxes_chunk = gt_bboxes.view({ -1, 1, 4 }).chunk(2, 2);
    auto lt = gt_bboxes_chunk[0];
    auto rb = gt_bboxes_chunk[1];
    // std::cout << "xy_centers " << xy_centers.sizes() << std::endl;
    auto xy_tmp = xy_centers.unsqueeze(0);
    // std::cout << "xy_center[None]" << xy_centers.unsqueeze(0).sizes() << "  "
    //     << xy_centers.index({torch::indexing::None}).sizes() << std::endl;
    auto x1y1 = xy_tmp - lt;
    auto x2y2 = rb - xy_tmp;
    auto bbox_deltas = torch::cat({ x1y1, x2y2 }, 2).view({ bs_, n_boxes, n_anchors, -1 });

    auto ret = bbox_deltas.amin(3);
    ret.gt_(eps);

    return ret;
}

torch::Tensor TaskAlignedAssignerImpl::select_topk_candidates(torch::Tensor metrics, bool largest /*= true*/,
    c10::optional<torch::Tensor> topk_mask /*= c10::nullopt*/)
{
    auto [topk_metrics, topk_idxs] = torch::topk(metrics, this->topk, -1, largest);

    if (!topk_mask.has_value())
    {
        auto [topk_metrics_mv, topk_metrics_mi] = topk_metrics.max(-1, true);
        auto mask = (topk_metrics_mv > this->eps).expand_as(topk_idxs);
        topk_mask = mask;
    }

    topk_idxs.masked_fill_(torch::logical_not(topk_mask.value()), 0);

    auto count_tensor = torch::zeros(metrics.sizes(),
        torch::TensorOptions().dtype(torch::kInt8).device(topk_idxs.device()));
    auto t_ones = torch::ones_like(topk_idxs.index({
        torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1) }), 
        torch::TensorOptions().dtype(torch::kInt8).device(topk_idxs.device()));
    
    //  # Expand topk_idxs for each value of k and add 1 at the specified positions
    for (int k = 0; k < this->topk; k++)
        count_tensor.scatter_add_(-1, topk_idxs.index({
        torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(k, k + 1)
            }), t_ones);

    auto count_tensor_mask = count_tensor > 1;
    count_tensor.masked_fill_(count_tensor_mask, 0);

    torch::Tensor ret = count_tensor.to(metrics.dtype());
    return ret;
}   

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
TaskAlignedAssignerImpl::select_highest_overlaps(torch::Tensor mask_pos, torch::Tensor overlaps, int _n_max_boxes)
{
    auto fg_mask = mask_pos.sum({ -2 });
    if (fg_mask.max().item().toInt() > 1)
    {
        // # (b, n_max_boxes, h*w)
        auto tmp_fg_gt1 = fg_mask.unsqueeze(1) > 1;
        auto mask_multi_gts =tmp_fg_gt1.expand({ -1, _n_max_boxes, -1 });
        auto max_overlaps_idx = overlaps.argmax(1); // # (b, h*w)

        auto is_max_overlaps = torch::zeros(mask_pos.sizes(),
            torch::TensorOptions().dtype(mask_pos.dtype()).device(mask_pos.device()));
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1);

        mask_pos = torch::where(mask_multi_gts, is_max_overlaps, mask_pos).to(torch::kFloat);
        fg_mask = mask_pos.sum({ -2 });
    }

    auto target_gt_idx = mask_pos.argmax(-2);   // (b, h*w)
    return std::make_tuple(target_gt_idx, fg_mask, mask_pos);
}

/*
    """
    Compute target labels, target bounding boxes, and target scores for the positive anchor points.

    Args:
        gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                            batch size and max_num_obj is the maximum number of objects.
        gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
        target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                anchor points, with shape (b, h*w), where h*w is the total
                                number of anchor points.
        fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                            (foreground) anchor points.

    Returns:
        (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
            - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                        positive anchor points.
            - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                        for positive anchor points.
            - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                        for positive anchor points, where num_classes is the number
                                        of object classes.
    """
*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
TaskAlignedAssignerImpl::get_target(torch::Tensor gt_labels,
    torch::Tensor gt_bboxes,
    torch::Tensor target_gt_idx,
    torch::Tensor fg_mask)
{
    // Assigned target labels, (b, 1)
    auto batch_ind = torch::arange(this->bs,
        torch::TensorOptions().dtype(torch::kInt64).device(gt_labels.device())).index({"...", torch::indexing::None});
    target_gt_idx = target_gt_idx + batch_ind * this->n_max_boxes;
    auto target_labels = gt_labels.to(torch::kLong).flatten().index({ target_gt_idx });

    // # Assigned target boxes, (b, max_num_obj, 4) -> (b, h * w, 4)
    auto target_bboxes = gt_bboxes.view({ -1, gt_bboxes.size(-1) }).index({ target_gt_idx });
    target_labels.clamp_(0);
    //std::cout << "target_labels: " << target_labels.sizes() << std::endl;
    //  # 10x faster than F.one_hot() (b, h*w, 80)
    torch::Tensor target_scores = torch::zeros({
        target_labels.size(0), target_labels.size(1), this->num_classes },
        torch::TensorOptions().dtype(torch::kInt64).device(target_labels.device()));
    target_scores.scatter_(2, target_labels.unsqueeze(-1), 1);

    auto fg_scores_mask = fg_mask.index({torch::indexing::Slice(), 
        torch::indexing::Slice(), torch::indexing::None}
        ).repeat({1, 1, this->num_classes});
    auto fg_scores_mask_bool = fg_scores_mask.gt(0);
    target_scores = torch::where(fg_scores_mask > 0, target_scores, 0);
    //std::cout << "target_scores " << target_scores.sizes() << std::endl;
    return std::make_tuple(target_labels, target_bboxes, target_scores);
}
