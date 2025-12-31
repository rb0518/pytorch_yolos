#include "nms.h"

#include <opencv2/opencv.hpp>
#include "ops.h"
#include "utils.h"

std::vector<torch::Tensor> non_max_suppression(
    torch::Tensor prediction,
    float conf_thres/* = 0.25f*/,
    float iou_thres/* = 0.45f*/,
    std::vector<int> classes/* = {}*/,
    bool agnostic/* = false*/,
    bool multi_label/* = false*/,
    std::vector<int>/* = {}*/, // 暂时未用，用std::vector<int>代替
    int max_det/* = 300*/,
    int nc/* = 0*/,
    float max_time_img/* = 0.05f*/,
    int max_nms/* = 30000*/,
    int max_wh/* = 7680*/,
    bool in_place/* = true*/,
    bool rotated /*= false*/)
{
    if (prediction.size(-1) == 6)
    {
        std::cout << "prediction size(-1) == 6, it's end2ent model i.e. [1,300, 6] \n";
        std::cout << ColorString("ERROR: ", "R") << "not support now.\n";
        exit(-1);
    }

    int bs = prediction.size(0);    // batch size
    if (nc == 0) nc = prediction.size(1) - 4;
    int nm = prediction.size(1) - nc - 4;   // number of masks
    int mi = 4 + nc;    // mask start index

    auto xc = prediction.index({ torch::indexing::Slice(), torch::indexing::Slice(4, mi) }).amax(1) > conf_thres; // condidates
 
    float time_limit = 2.0 + max_time_img * bs; // seconds to quit after
    multi_label &= nc > 1;
    prediction = prediction.transpose(-1, -2);

    if(!rotated)
    { 
        if (in_place)
        {
            auto pred_bbox_xyxy = xywh2xyxy(prediction.index(
                { "...", torch::indexing::Slice(0, 4) }));
            prediction.index_put_({ "...", torch::indexing::Slice(0,4)},
                pred_bbox_xyxy);
        }
        else
        { 
            auto xyxy_1 = xywh2xyxy(prediction.index({ "...",
                torch::indexing::Slice(0, 4) }));
            prediction = torch::cat({ xyxy_1,
                prediction.index({"...", torch::indexing::Slice(4, torch::indexing::None)}) }
                , - 1);
        }
    }

    std::vector<torch::Tensor> output;
    for (int i = 0; i < bs; i++)
        output.emplace_back(torch::zeros({ 0, 6 + nm },
            torch::TensorOptions().device(prediction.device())));

    for (size_t xi = 0; xi < prediction.size(0); xi++)
    {
        torch::Tensor x = prediction[xi];
        x = x.index({ xc[xi] });
        if (x.size(0) == 0)
        {
            continue;
        }

        auto xyxy_cls_mask = x.split({ 4, nc, nm }, 1);
        torch::Tensor box = xyxy_cls_mask[0];
        torch::Tensor cls = xyxy_cls_mask[1];
        torch::Tensor mask = xyxy_cls_mask[2];

        if (multi_label)
        {
            auto cls_where = torch::where(cls > conf_thres);
            torch::Tensor i = cls_where[0];
            torch::Tensor j = cls_where[1];

            auto box_i = box.index({ i });
            auto x_ij = x.index({ i, 4 + j, torch::indexing::None });
            auto mask_i = mask.index({ i });
            auto j_float = j.index({torch::indexing::Slice(), 
                torch::indexing::None});
            j_float = j_float.to(torch::kFloat);
            x = torch::cat({ box_i, x_ij, j_float, mask_i }, 1);
        }
        else
        {
            torch::Tensor conf, j;
            std::tie(conf, j) = at::_ops::max_dim::call(cls, 1, true);
            x = torch::cat({ box, conf, j.to(torch::kFloat), mask }, 1).index(
                {conf.view(-1)> conf_thres});
        }

        if (classes.size())
        {
            // not complete code
        }

        int n = x.size(0);   // number of boxes

        if (n < 1)
            continue;

        if (n > max_nms) 
        { 
            // 获取第4列（置信度）并按降序排序，取前max_nms个索引
            auto x_argsort = x.index({torch::indexing::Slice(), 4 }).argsort(-1, true);
            x_argsort = x_argsort.index({ torch::indexing::Slice(0, max_nms) });
            x = x.index({ x_argsort });
        }

        torch::Tensor c;
        if (agnostic)
            c = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, 6) }) * 0;

        else
            c = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, 6) }) * max_wh;
        auto scores = x.index({ torch::indexing::Slice(), 4 });

#if 1
        auto boxes = x.index({ torch::indexing::Slice(), torch::indexing::Slice(
            0, 4) }).add(c);
        
        auto nms_indices = vision::ops::nms(boxes, scores, iou_thres);
        nms_indices = nms_indices.index({torch::indexing::Slice(0, max_det)});

        output[xi] = x.index({ nms_indices }).clone();

#else
        std::vector<int> nms_indices;
        if (rotated)
        {
            std::cout << "rotated not support now. \n";

        }
        else
        {
            auto boxes = x.index({ torch::indexing::Slice(), torch::indexing::Slice(
                0, 4) }).add(c);
            
            std::vector<cv::Rect> cv_boxes;
            std::vector<float> cv_scores;
            for (int i = 0; i < boxes.size(0); i++)
            {
                auto x1 = boxes[i][0].item().toDouble();
                auto y1 = boxes[i][1].item().toDouble();
                auto x2 = boxes[i][2].item().toDouble();
                auto y2 = boxes[i][3].item().toDouble();

                cv_boxes.push_back(cv::Rect(cv::Point2d(x1, y1), cv::Point2d(x2, y2)));
                cv_scores.push_back(scores[i].item().toFloat());
            }

            // run cv::dnn::NMSBoxes
#if 1
            std::vector<float> updated_scores;
            cv::dnn::softNMSBoxes(cv_boxes, cv_scores, updated_scores, conf_thres, iou_thres, nms_indices);
#else
            cv::dnn::NMSBoxes(cv_boxes, cv_scores, conf_thres, iou_thres, nms_indices);
#endif   
        }
        if (nms_indices.size() > max_det) {
            nms_indices.resize(max_det);
        }
        output[xi] = x.index({ torch::tensor(nms_indices) });
#endif        
    }

    return output;
}

namespace vision{
namespace ops{

at::Tensor nms_kernel_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) 
{
    TORCH_CHECK(dets.is_cpu(), "dets must be a CPU tensor");
    TORCH_CHECK(scores.is_cpu(), "scores must be a CPU tensor");
    TORCH_CHECK(
        dets.scalar_type() == scores.scalar_type(),
        "dets should have the same type as scores");
    
    std::cout << "dets.numel: " << dets.numel() << std::endl;
    if (dets.numel() == 0) {
        return at::empty({ 0 }, dets.options().dtype(at::kLong));
    }

    auto x1_t = dets.select(1, 0).contiguous();
    auto y1_t = dets.select(1, 1).contiguous();
    auto x2_t = dets.select(1, 2).contiguous();
    auto y2_t = dets.select(1, 3).contiguous();

    at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    auto order_t = std::get<1>(
        scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

    auto ndets = dets.size(0);
    at::Tensor suppressed_t = at::zeros({ ndets }, dets.options().dtype(at::kByte));
    at::Tensor keep_t = at::zeros({ ndets }, dets.options().dtype(at::kLong));
    auto scalar_t = dets.scalar_type();
    auto suppressed = suppressed_t.data_ptr<uint8_t>();
    auto keep = keep_t.data_ptr<int64_t>();
    auto order = order_t.data_ptr<int64_t>();
    auto x1 = x1_t.data_ptr<float>();
    auto y1 = y1_t.data_ptr<float>();
    auto x2 = x2_t.data_ptr<float>();
    auto y2 = y2_t.data_ptr<float>();
    auto areas = areas_t.data_ptr<float>();

    int64_t num_to_keep = 0;

    for (int64_t _i = 0; _i < ndets; _i++) {
        auto i = order[_i];
        if (suppressed[i] == 1) {
            continue;
        }
        keep[num_to_keep++] = i;
        auto ix1 = x1[i];
        auto iy1 = y1[i];
        auto ix2 = x2[i];
        auto iy2 = y2[i];
        auto iarea = areas[i];

        for (int64_t _j = _i + 1; _j < ndets; _j++) {
            auto j = order[_j];
            if (suppressed[j] == 1) {
                continue;
            }
            auto xx1 = std::max(ix1, x1[j]);
            auto yy1 = std::max(iy1, y1[j]);
            auto xx2 = std::min(ix2, x2[j]);
            auto yy2 = std::min(iy2, y2[j]);

            auto w = std::max(0.f, xx2 - xx1);
            auto h = std::max(0.f, yy2 - yy1);
            auto inter = w * h;
            auto ovr = inter / (iarea + areas[j] - inter);
            if (ovr > iou_threshold) {
                suppressed[j] = 1;
            }
        }
    }
    return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

   
torch::Tensor nms(
    const torch::Tensor& dets,
    const torch::Tensor& scores,
    double iou_threshold)
{
    TORCH_CHECK(
        dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
    TORCH_CHECK(
        dets.size(1) == 4,
        "boxes should have 4 elements in dimension 1, got ",
        dets.size(1));
    TORCH_CHECK(
        scores.dim() == 1,
        "scores should be a 1d tensor, got ",
        scores.dim(),
        "D");
    TORCH_CHECK(
        dets.size(0) == scores.size(0),
        "boxes and scores should have same number of elements in ",
        "dimension 0, got ",
        dets.size(0),
        " and ",
        scores.size(0));
    
    auto result = at::empty({ 0 }, dets.options());

    if (dets.is_cuda())
    {
        result = nms_kernel_cuda(dets, scores, iou_threshold);
    }
    else
    {
        result = nms_kernel_cpu(dets, scores, iou_threshold);
    }

    return result;
}

}   //namespace ops
}   //namespace vision

