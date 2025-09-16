#include <torch/torch.h>
#include <filesystem>
#include <fstream>

#include "metrics.h"

#include "general.h"
// Model fitness as a weighted combination of metrics
torch::Tensor fitness(torch::Tensor x)
{
	// weights for [P, R, mAP@0.5, mAP@0.5:0.95]
	std::vector<float> w = { 0.0f, 0.0f, 0.1f, 0.9f };
	torch::Tensor w_t = torch::tensor(w);
	auto ret = x.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4) }) * w_t;
	return ret.sum(1);
}

// 功能，tensor，去重并排序
torch::Tensor torch_unique(const torch::Tensor& input) 
{
    auto cpu_tensor = input.cpu().contiguous();
    auto accessor = cpu_tensor.accessor<float, 1>();
    std::vector<float> vec(accessor.data(), accessor.data() + cpu_tensor.numel());

    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());

    return torch::from_blob(vec.data(), { static_cast<int64_t>(vec.size()) },
        torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// 返回索和计数，支持逆向索引
std::tuple<torch::Tensor, torch::Tensor> /*, torch::Tensor, torch::Tensor>*/
torch_unique_full(const torch::Tensor& input, bool return_index = true,
    bool return_inverse = false, bool return_counts = false) {
    auto cpu_tensor = input.cpu().contiguous();
    auto accessor = cpu_tensor.accessor<float, 1>();
    const int64_t size = cpu_tensor.numel();

    // 存储值与索引值
    std::vector<std::pair<float, int64_t>> pairs;
    for (int64_t i = 0; i < size; ++i) {
        pairs.emplace_back(accessor[i], i);
    }

    // 排序和去重
    std::sort(pairs.begin(), pairs.end());
    auto last = std::unique(pairs.begin(), pairs.end(),
        [](const auto& a, const auto& b) { return a.first == b.first; });
    pairs.erase(last, pairs.end());

    // 提取唯一值 
    std::vector<float> unique_values;
    std::vector<int64_t> indices, inverse(size), counts;
    for (const auto& p : pairs) {
        unique_values.push_back(p.first);
    }

    // 计算逆向索引和计数
    std::unordered_map<float, int64_t> value_to_index;
    for (int64_t i = 0; i < pairs.size(); ++i) {
        value_to_index[pairs[i].first] = i;
    }

    std::unordered_map<float, int64_t> value_counts;
    for (int64_t i = 0; i < size; ++i) {
        float val = accessor[i];
        inverse[i] = value_to_index[val];
        value_counts[val]++;
    }

    // 根据输入变量，准备返回数据
    torch::Tensor unique_tensor = torch::from_blob(
        unique_values.data(), { static_cast<int64_t>(unique_values.size()) },
        torch::TensorOptions().dtype(torch::kFloat32)).clone();

    torch::Tensor indices_tensor, inverse_tensor, counts_tensor;

    if (return_index) {
        std::vector<int64_t> indices;
        for (const auto& p : pairs) indices.push_back(p.second);
        indices_tensor = torch::from_blob(
            indices.data(), { static_cast<int64_t>(indices.size()) },
            torch::kInt64).clone();
    }

    if (return_inverse) {
        inverse_tensor = torch::from_blob(
            inverse.data(), { size }, torch::kInt64).clone();
    }

    if (return_counts) {
        std::vector<int64_t> counts;
        for (const auto& p : pairs) counts.push_back(value_counts[p.first]);
        counts_tensor = torch::from_blob(
            counts.data(), { static_cast<int64_t>(counts.size()) },
            torch::kInt64).clone();
    }

    return std::make_tuple(unique_tensor, indices_tensor); // , inverse_tensor, counts_tensor);
}

/*
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
*/
torch::Tensor process_batch(torch::Tensor& detections, torch::Tensor& labels, torch::Tensor& iouv)
{
    //std::cout << "detections: " << detections.device().type() << " labels: " << labels.device().type() << " iouv: " << iouv.device().type() << std::endl;
    torch::Tensor correct = torch::zeros({detections.size(0), iouv.size(0)}, 
                                        torch::dtype(torch::kBool)).to(iouv.device());
    
    auto iou = box_iou(labels.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}),
                        detections.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 4)})).to(iouv.device());
    auto x = torch::where((iou >= iouv[0]) & (
        labels.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)}) ==
        detections.index({torch::indexing::Slice(), 5})));

    //std::cout << "process_batch x " << x.size() << std::endl;
    if(x[0].size(0))
    {
        auto x_stack = torch::stack(x, 1);
        //std::cout << x_stack.device().type() << " " << x_stack.sizes() << std::endl;
        auto iou_ind = iou.index({ x[0], x[1] }).index({ torch::indexing::Slice(), torch::indexing::None });
        //std::cout << iou_ind.sizes() << std::endl;

        auto matches = torch::cat({ x_stack, iou_ind }, 1).to(iouv.device());
        //std::cout << "matches: " << matches.sizes() << " x0.shape[0] " << x[0].size(0) << std::endl;
        if (x[0].size(0) > 1)
        {
            // python 原码：matches = matches[matches[:, 2].argsort()[::-1]]
            auto sorted_indices = torch::argsort(matches.index({ torch::indexing::Slice(), 2 }));
            sorted_indices = torch::flip(sorted_indices, /*dim=*/0);
            matches = matches.index({ sorted_indices });
            //  matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

            auto col1 = matches.index({ torch::indexing::Slice(), 1 });
            auto unique_result = torch::_unique2(col1, true, false, false);
            auto unique_indices = std::get<1>(unique_result);
            matches = matches.index_select(0, unique_indices);
//            std::cout << "matches: " << matches.sizes() << std::endl;
            //  matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            auto col2 = matches.index({ torch::indexing::Slice(), 0 });
            unique_result = torch::_unique2(col2, true, false, false);
            unique_indices = std::get<1>(unique_result);
            matches = matches.index_select(0, unique_indices);
        }

        matches = matches.to(iouv.device());
        auto iouv_filter = matches.index({ torch::indexing::Slice(), torch::indexing::Slice(2, 3) }) >= iouv;
        //std::cout << matches.sizes() << " iou_filter: " << iouv_filter.sizes() << std::endl;
        matches = matches.to(torch::kLong);
        correct.index_put_({ matches.index({torch::indexing::Slice(), 1}) }, iouv_filter);
    }
    return correct;
}

// 计算平均精确度(AP)
std::tuple<double, torch::Tensor, torch::Tensor> compute_ap(
    const torch::Tensor& recall,
    const torch::Tensor& precision,
    const std::string& method/* = "interp"*/)
{
    // 数据转换到CPU上
    auto r = recall.cpu().to(torch::kFloat32).contiguous();
    auto p = precision.cpu().to(torch::kFloat32).contiguous();

    // 输入维度有效性判定
    if (r.dim() != 1 || p.dim() != 1) {
        throw std::invalid_argument("recall and precision must be 1D tensors");
    }
    if (r.numel() != p.numel()) {
        throw std::invalid_argument("recall and precision must have the same length");
    }
    if (r.numel() == 0) {
        return std::make_tuple(0.0, torch::tensor({ 1.0f, 0.0f }), torch::tensor({ 0.0f, 0.01f }));
    }

    
    float last_recall = r[-1].item<float>();
    auto mrec = torch::cat({
        torch::tensor({0.0f}, torch::kFloat32),  // 头部添加0.0
        r,                                       // 原始召回率
        torch::tensor({last_recall + 0.01f}, torch::kFloat32)  // 结束添加recall[-1]+0.01
        });

    auto mpre = torch::cat({
        torch::tensor({1.0f}, torch::kFloat32),  // 开头添加1.0
        p,                                       // 原始精确率
        torch::tensor({0.0f}, torch::kFloat32)   // 尾部添加0.0
        });

    // 计算精确率 (取最大值)
    auto flipped_mpre = torch::flip(mpre, { 0 });  // 翻转
    auto accumulated = std::get<0>(torch::cummax(flipped_mpre, 0));  // 最大值累计
    mpre = torch::flip(accumulated, { 0 });  

    double ap = 0.0;

    if (method == "interp") {
        // 线性插值
        auto x = torch::linspace(0.0f, 1.0f, 101);  // 共计101个值
        auto interp_mpre = torch::from_blob(
            new float[101],
            { 101 },
            torch::kFloat32
        ).set_requires_grad(false);

        for (int i = 0; i < 101; ++i) {
            float xi = x[i].item<float>();
            int idx = 0;
            while (idx < mrec.numel() && mrec[idx].item<float>() < xi) {
                idx++;
            }

            if (idx == 0) {
                interp_mpre[i] = mpre[0].item<float>();
            }
            else if (idx >= mrec.numel()) {
                interp_mpre[i] = mpre[-1].item<float>();
            }
            else {
                float x0 = mrec[idx - 1].item<float>();
                float x1 = mrec[idx].item<float>();
                float y0 = mpre[idx - 1].item<float>();
                float y1 = mpre[idx].item<float>();
                interp_mpre[i] = y0 + (y1 - y0) * (xi - x0) / (x1 - x0 + 1e-16f);
            }
        }

        // 对应 (np.trapz)
        ap = 0.0;
        for (int i = 1; i < 101; ++i) {
            ap += (x[i].item<float>() - x[i - 1].item<float>()) *
                (interp_mpre[i].item<float>() + interp_mpre[i - 1].item<float>()) / 2.0f;
        }
       
        delete[] interp_mpre.data_ptr<float>();
    }
    else {  // method = 'continuous'
        std::vector<int64_t> indices;
        for (int i = 1; i < mrec.numel(); ++i) {
            if (std::abs(mrec[i].item<float>() - mrec[i - 1].item<float>()) > 1e-6f) {
                indices.push_back(i - 1);  
            }
        }

        for (int64_t i : indices) {
            ap += (mrec[i + 1].item<float>() - mrec[i].item<float>()) *
                mpre[i + 1].item<float>();
        }
    }

    return std::make_tuple(ap, mpre, mrec);
}

std::vector<float> tensor_to_vector(const torch::Tensor& tensor, bool neg = false) 
{
    // 确保张量是连续且为float类型
    torch::Tensor contiguous_tensor = tensor.contiguous().to(torch::kFloat32);
    // 通过指针直接初始化vector
    if (neg)
        contiguous_tensor = 0 - contiguous_tensor;

    return std::vector<float>(
        contiguous_tensor.data_ptr<float>(),
        contiguous_tensor.data_ptr<float>() + contiguous_tensor.numel()
    );
}

/*
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

        int num_samples = 100;
        int num_iou_thresholds = 10;

        auto tp = torch::randint(0, 2, {num_samples, num_iou_thresholds}, torch::kFloat32);
        auto conf = torch::rand({num_samples});
        auto pred_cls = torch::randint(0, 5, {num_samples}, torch::kInt64);  // 5�����
        auto target_cls = torch::randint(0, 5, {num_samples}, torch::kInt64);

        // 返回数据内容定义：
        auto [p, r, ap, f1, classes] = ap_per_class(tp, conf, pred_cls, target_cls, false);

*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ap_per_class(
    const torch::Tensor& tp,
    const torch::Tensor& conf,
    const torch::Tensor& pred_cls,
    const torch::Tensor& target_cls,
    bool plot/* = false*/,
    const std::string& save_dir /*= "."*/,
    const std::vector<std::string>& names /*= {}*/)
{
    auto tp_cpu = tp.cpu().to(torch::kFloat32);
    auto conf_cpu = conf.cpu().to(torch::kFloat32);
    auto pred_cls_cpu = pred_cls.cpu().to(torch::kFloat32);
    auto target_cls_cpu = target_cls.cpu().to(torch::kFloat32);

    // std::cout << "\n tp: " << tp.sizes() << "conf: " << conf.sizes() << " pred:" << pred_cls.sizes() 
    //     << "target_cls: " << target_cls.sizes() << std::endl; 

    auto [sorted_conf, sorted_indices] = torch::sort(conf_cpu, 0, true);
    auto tp_sorted = tp_cpu.index_select(0, sorted_indices);
    auto conf_sorted = conf_cpu.index_select(0, sorted_indices);
    auto pred_cls_sorted = pred_cls_cpu.index_select(0, sorted_indices);
    //std::cout << " after sort tp: " << tp_sorted.sizes() << "conf: " << conf_sorted.sizes() << " pred:" << pred_cls_sorted.sizes(); 

    // 去重操作
    auto unique_classes = torch_unique(target_cls_cpu);
    int nc = unique_classes.size(0);  

    torch::Tensor px = torch::linspace(0.0f, 1.0f, 1000);
    // py = []
    torch::Tensor ap = torch::zeros({ nc, tp.size(1) }, torch::kFloat32);
    torch::Tensor p = torch::zeros({ nc, 1000 }, torch::kFloat32);
    torch::Tensor r = torch::zeros({ nc, 1000 }, torch::kFloat32);

    //std::cout << "\n ap_per_class : nc : " << nc << " unique_classes " << unique_classes.sizes() << std::endl;

    for (int ci = 0; ci < nc; ++ci) {
        int c = unique_classes[ci].item().toInt();
        auto mask = (pred_cls_sorted == c);
        auto target_mask = (target_cls_cpu == c);
        //std::cout << ci <<": " << c <<" mask: " << mask.sizes() << " target_mask: " << target_mask.sum() << std::endl;
        int n_p = mask.sum().item().toInt();  
        int n_l = (target_cls_cpu == c).sum().item().toInt();  
        //std::cout << n_p << " " << n_l << std::endl;
        if (n_p == 0 || n_l == 0) {
            continue;
        }

        auto tp_i = tp_sorted.index({ mask });
        //std::cout << "tp_i: " << tp_i.sizes() << std::endl; 
        auto fpc = (1 - tp_i).cumsum(0);
        //std::cout << "fpc: " << fpc.sizes() << std::endl;
        auto tpc = tp_i.cumsum(0);
        //std::cout << " tpc: " << tpc.sizes() << std::endl;

        auto linear_interp = [](float x, const std::vector<float>& xp, const std::vector<float>& fp, float f_left = 0) {
            auto it = std::lower_bound(xp.begin(), xp.end(), x);
            if (it == xp.begin()) return f_left;
            if (it == xp.end()) return fp.back();

            size_t xp_idx = it - xp.begin();    //在那两点间，用这两个坐标计算直线斜率，插值得到返回y坐标 
            float x0 = xp[xp_idx - 1], x1 = xp[xp_idx];
            float f0 = fp[xp_idx - 1], f1 = fp[xp_idx];
            return f0 + (x - x0) * (f1 - f0) / (x1 - x0);
            };

        auto conf_indexmask = 0.f - conf_sorted.index({ mask });

        auto recall = tpc / (n_l + 1e-16f);
        auto precision = tpc / (tpc + fpc + 1e-16f);

        //std::cout << "recall " << recall.sizes() << " precision: " << precision.sizes() << std::endl;
        auto conf_i = conf_sorted.index({ mask });
        //std::cout << "conf_i: " << conf_i.sizes() << std::endl;
        conf_i = 0 - conf_i;
        std::vector<float> conf_negs_v = tensor_to_vector(conf_i);
        //std::cout << "recall [0] " << recall.index({ torch::indexing::Slice(), 0 }).sizes() << std::endl;
        //std::cout << "recall [0] " << precision.index({ torch::indexing::Slice(), 0 }).sizes() << std::endl;
        std::vector<float> recall_v = tensor_to_vector(recall.index({ torch::indexing::Slice(), 0 }));
        std::vector<float> precision_v = tensor_to_vector(recall.index({ torch::indexing::Slice(), 0 }));

        for (int px_i = 0; px_i < px.size(0); px_i++)
        {
            auto v_px = px[px_i].item().toFloat();
            auto interp_r_tmp = linear_interp(-v_px, conf_negs_v, recall_v, 0.f);
            r.index_put_({ ci, px_i }, interp_r_tmp);

            auto interp_p_tmp = linear_interp(-v_px, conf_negs_v, precision_v, 1.f);
            p.index_put_({ ci, px_i }, interp_p_tmp);
        }

        //std::cout << "ci " << ci ;
        for (int j = 0; j < tp.size(1); ++j) {
            auto [ap_val, mprc, mrec] = compute_ap(recall.index({ torch::indexing::Slice(), j }),
                precision.index({ torch::indexing::Slice(), j}), "interp");
            ap[ci][j] = ap_val;
        }
        //std::cout << " compute_ap over." << std::endl;
    }

    torch::Tensor f1 = 2 * p * r / (p + r + 1e-16f);

    auto f1_mean = f1.mean(0);
    //std::cout << "f1_mean " << f1_mean.sizes() << " " << f1_mean.argmax().sizes() << std::endl;
    int max_f1_idx = f1_mean.argmax().item().toInt();

    return std::make_tuple(
        p.index({ torch::indexing::Slice(), max_f1_idx }),
        r.index({ torch::indexing::Slice(), max_f1_idx }),
        ap,
        f1.index({ torch::indexing::Slice(), max_f1_idx }),
        unique_classes.to(torch::kInt32)
    );
}
/*
ConfusionMatrix::ConfusionMatrix(int nc, float conf, float iou_thres)
    : nc_(nc), conf_thres_(conf), iou_thres_(iou_thres) 
{
    matrix_ = torch::zeros({ nc_ + 1, nc_ + 1 }, torch::kInt64);
}

void ConfusionMatrix::process_batch(const torch::Tensor& detections, const torch::Tensor& labels) 
{
    // 转换到CPU
}

torch::Tensor ConfusionMatrix::get_matrix()
{
    return matrix_.clone();
}

void ConfusionMatrix::print(){
    std::cout << "Confusion Matrix (" << nc_ << "classes + background):\n";
}

// ???只保留函数名，与python代码保持一致
void ConfusionMatrix::plot(const std::string& save_dir, const std::vector<std::string>& names) 
{
}
*/