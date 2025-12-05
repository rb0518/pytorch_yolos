#include "nms.h"

#include <opencv2/opencv.hpp>
#include "ops.h"
#include "utils.h"

/*
    python 原代码函数注释内容
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
*/
std::vector<torch::Tensor> non_max_suppression(
    torch::Tensor prediction,
    float conf_thres,
    float iou_thres,
    const std::vector<std::vector<int>>& classes,
    bool agnostic,
    bool multi_label,
    const std::vector<std::vector<float>>& labels,
    int max_det /*= 300*/,
    int nm /*= 0*/  // number of masks
) 
{
    int bs = prediction.size(0);        // batch size
    int nc = prediction.size(2) - nm - 5;  // number of classes
    auto xc = prediction.index({"...", 4}) > conf_thres; // candidates
    //std::cout << "input: " << prediction.sizes() << " " << xc.sizes() << std::endl;

    // 参数初始化
    //constexpr int min_wh = 2;
    constexpr int max_wh = 7680;    // 10-9 4096==>7680
    constexpr int max_nms = 30000;
    constexpr float time_limit = 30.0;
    bool redundant = true;
    bool merge = false;

    multi_label &= (nc > 1);

    auto t = std::chrono::steady_clock::now();

    int mi = 5 + nc;    // mask start index

    std::vector<torch::Tensor> output;
    // auto output = std::vector<torch::Tensor>(prediction.size(0), torch::zeros({ 0, 6 }, prediction.options()));

    // 主处理循环
    for (int xi = 0; xi < prediction.size(0); ++xi) 
    {
        auto x = prediction[xi];
//        std::cout << xc[xi].sizes() << xc[xi].sum().item().toInt() << std::endl;

        // 应用置信度筛选
        x = x.index({ xc[xi] });
        //std::cout << xi << " x " << x.sizes() << std::endl;
        // ???添加先验标签，先不处理
        /*
        if (labels.size() > 0 && labels[xi].size() > 0) {
            auto l = torch::from_blob(labels[xi].data(), { labels[xi].size(), 5 }, torch::kFloat32);
            auto v = torch::zeros({ l.size(0), x.size(1) }, x.options());
            v.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4) }) = l.index({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) });
            v.index({ torch::indexing::Slice(), 4 }) = 1.0;
            v.index({ torch::indexing::Slice(0, l.size(0)), torch::indexing::Slice(5, x.size(1)) }) = l.index({ torch::indexing::Slice(), torch::indexing::Slice(0, x.size(1) - 5) });
            x = torch::cat({ x, v }, 0);
        }
        */
        // 如果无剩余框则跳过
        if (x.size(0) == 0)
        {
//            std::cout << xi << " after x[xc[xi]] return 0 " << std::endl;
            output.push_back(torch::zeros({0, 6 + nm}));
            continue;
        }

        // 计算最终置信度
        x.index_put_(
            { torch::indexing::Slice(), torch::indexing::Slice(5, torch::indexing::None)},
            x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, torch::indexing::None)}) * x.index({ torch::indexing::Slice(), torch::indexing::Slice(4, 5)})
            );

        // 坐标转换
        auto box = xywh2xyxy(x.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4) }));
        // zero columns if no masks
        auto mask = x.index({torch::indexing::Slice(), torch::indexing::Slice(mi, torch::indexing::None)});
        // std::cout << "box " << box.sizes() << "multi_label: " << multi_label <<" mask: " << mask.sizes() << std::endl;

        // 多标签处理
        if (multi_label) {
            auto ijcls = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, mi) }) > conf_thres;
            auto ijcls_t = ijcls.nonzero().t();
            auto i = ijcls_t[0];
            auto j = ijcls_t[1];
            auto box_i = box.index({ i });
            auto x_ij = x.index({i, j+5}).unsqueeze(-1);
            auto mask_i = mask.index({i});
            std::cout << "box_i: " << box_i.sizes() << " mask_i " << mask_i.sizes() << std::endl;
            x = torch::cat({ box_i, x_ij, j.unsqueeze(-1).to(torch::kFloat), mask_i}, 1);
        }
        else {
            auto [conf, j] = x.index({ torch::indexing::Slice(), torch::indexing::Slice(5, mi) }).max(1, true);
            auto x_filtered = torch::cat({ box, conf, j.to(torch::kFloat), mask}, 1);
            auto conf_sel = conf > conf_thres;
//            std::cout << "x_filtered: " << x_filtered.sizes() << " conf " << conf_sel.sizes() << std::endl;
            x = x_filtered.index({conf_sel.squeeze(1)});
        }

        // 按类别筛选
        // if (!classes.empty()) {
        //     auto mask = torch::zeros({ x.size(0), 1 }, x.options());
        //     for (auto cls : classes[xi]) {
        //         mask |= (x.index({ torch::indexing::Slice(), 5 }) == cls).view(-1, 1);
        //     }
        //     x = x[mask];
        // }
        if(x.size(0) ==  0)
        {
//            std::cout << " no boxes " << std::endl;
            output.push_back(torch::zeros({0, 6 + nm}));
            continue;
        }

        // 应用限制条件
        if (x.size(0) > max_nms) {
            x = x[x.index({ torch::indexing::Slice(), 4 }).argsort(0, true).slice(0, max_nms)];
        }
//        std::cout << "before nms x " << x.sizes() << std::endl;
        // batched nms
        auto x_cpu = x.cpu();

        // Batched NMS
        auto c = x.index({torch::indexing::Slice(), 5});
        if(agnostic)
            c = c * 0;
        else 
            c = c * max_wh;
        
        auto boxes = x.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)}) + c.unsqueeze(1);
        auto scores = x.index({torch::indexing::Slice(), 4});

        std::vector<cv::Rect> cv_boxes;
        std::vector<float> cv_scores;
        for(int i = 0; i < x.size(0); i++)
        {
            auto x1 = boxes[i][0].item().toDouble();
            auto y1 = boxes[i][1].item().toDouble();
            auto x2 = boxes[i][2].item().toDouble();
            auto y2 = boxes[i][3].item().toDouble();

            cv_boxes.push_back(cv::Rect(cv::Point2d(x1, y1), cv::Point2d(x2, y2)));
            cv_scores.push_back(scores[i].item().toFloat());
        }

        // run cv::dnn::NMSBoxes
        std::vector<int> nms_indices;
#if 0
        std::vector<float> updated_scores;
        cv::dnn::softNMSBoxes(cv_boxes, cv_scores, updated_scores, conf_thres, iou_thres, nms_indices);
#else
        cv::dnn::NMSBoxes(cv_boxes, cv_scores, conf_thres, iou_thres, nms_indices);
#endif        
//        std::cout << "After cv::dnn::NMSBoxes: " << nms_indices.size() << std::endl;

        // 构造返回函数，压入vector
        int box_count = nms_indices.size();
        if (nms_indices.size() > 0)
        {
            auto box_sel = torch::tensor(nms_indices);
            auto ret_box = x_cpu.index({box_sel});
            output.push_back(ret_box);

            std::cout << xi << " return " << ret_box.sizes() << std::endl;
        }
        else
        {
            output.push_back(torch::zeros({0, 6 + nm}));
        }
 
        // 检查时间限制
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t).count() > time_limit) {
            std::cout << "WARNING: NMS time limit " << time_limit << "s exceeded" << std::endl;
            break;
        }
    }

    return output;
}

namespace ops {
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
    std::cout << "non_max_suppressio start, input: " << prediction.sizes() << std::endl;

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
    std::cout << "xc " << xc.sizes() << std::endl;      // [bs, 8400]
    float time_limit = 2.0 + max_time_img * bs; // seconds to quit after
    multi_label &= nc > 1;
    prediction = prediction.transpose(-1, -2);
    std::cout << "pred transpose: " << prediction.sizes() << "\n";

    if(!rotated)
    { 
        if (in_place)
        {
            auto pred_bbox_xyxy = xywh2xyxy(prediction.index(
                { "...", torch::indexing::Slice(0, 4) }));
            std::cout << "in_place pred_bbox_xyxy: " << pred_bbox_xyxy.sizes() << std::endl;
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

    for (size_t xi = 0; xi < prediction.size(0); ++xi)
    {
        torch::Tensor x = prediction[xi];
        x = x.index({ xc[xi] });
        if (x.size(0) == 0)
        {
            std::cout << xi << " after x[xc[xi]] return 0 " << std::endl;
            continue;
        }

        auto xyxy_cls_mask = x.split({ 4, nc, nm }, 1);
        torch::Tensor box = xyxy_cls_mask[0];
        torch::Tensor cls = xyxy_cls_mask[1];
        torch::Tensor mask = xyxy_cls_mask[2];
        std::cout << "cls " << cls.sizes() << " box " << box.sizes()
            << " mask " << mask.sizes() << std::endl;
        if (multi_label)
        {
            auto cls_where = torch::where(cls > conf_thres);
            torch::Tensor i = cls_where[0];
            torch::Tensor j = cls_where[1];
            std::cout << "i " << i.sizes() << " j " << j.sizes() << std::endl;
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
        std::cout << "number of boxes: " << n << std::endl;

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
#if 0
            std::vector<float> updated_scores;
            cv::dnn::softNMSBoxes(cv_boxes, cv_scores, updated_scores, conf_thres, iou_thres, nms_indices);
#else
            cv::dnn::NMSBoxes(cv_boxes, cv_scores, conf_thres, iou_thres, nms_indices);
#endif   
            if (nms_indices.size() > max_det) {
                nms_indices.resize(max_det);
            }
        }
        output[xi] = x.index({ torch::tensor(nms_indices) });
    }

    return output;
}

}   // end namespace ops