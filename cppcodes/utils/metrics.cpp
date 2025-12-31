#include <torch/torch.h>
#include <filesystem>
#include <fstream>

#include "metrics.h"
#include "general.h"

/*
 对应numpy.interp函数
*/
torch::Tensor torch_interp(const torch::Tensor& x, const torch::Tensor& xp, const torch::Tensor& fp, float left_v = 0.f) 
{
    // 确保是升序排列
    auto xp_indexes = torch::argsort(xp);
    auto xp_sorted = xp.index({ xp_indexes });
    auto fp_sorted = fp.index({ xp_indexes });

    // 得到x对应在xp上的位置
    auto indices = torch::searchsorted(xp_sorted, x, true) - 1;
    // 通过-1操作得到超左边界的点
    auto need_lefts = indices < 0;

    auto indices_clamped = torch::clamp(indices, 0, xp_sorted.size(0) - 2);
    auto x0 = xp_sorted.index({ indices_clamped });
    auto x1 = xp_sorted.index({ indices_clamped + 1 });
    auto y0 = fp_sorted.index({ indices_clamped });
    auto y1 = fp_sorted.index({ indices_clamped + 1 });
    auto t = (x - x0) / (x1 - x0);
    auto result = y0 + t * (y1 - y0);

    // 对超左边界的点用设定值替换
    result.index_put_({ need_lefts }, torch::full_like(result.index({ need_lefts }), left_v));
    return result;
}

torch::Tensor smooth(const torch::Tensor& y, float f = 0.05) 
{
    int nf = std::round(y.size(0) * f * 2) / 2 + 1;  // number of filter elements (must be odd) 奇数滤波器长度
    auto p = torch::ones({ nf / 2 }, y.options());  // 1填充
    auto yp = torch::cat({ p * y[0], y, p * y[y.size(0) - 1] });  // 边界填充
    auto kernel = torch::ones({ nf }, y.options()) / nf;  // 均值卷积核
    return torch::conv1d(yp.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), {}, 1, nf / 2).squeeze();  // 卷积平滑
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
        torch::tensor({1.0f}, torch::kFloat32)  // 结束添加recall[-1]+0.01
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

// 2025-12-19 return tuple(12 tensor)
std::tuple<
    torch::Tensor,  // tp
    torch::Tensor,  // fp
    torch::Tensor,  // p
    torch::Tensor,  // r
    torch::Tensor,  // f1

    torch::Tensor,  // ap
    torch::Tensor,  // unique_classes
    torch::Tensor,  // p_curve
    torch::Tensor,  // r_curve
    torch::Tensor,  // f1_curve

    torch::Tensor,  // x
    torch::Tensor   // prec_values
>
ap_per_class(
    const torch::Tensor& tp_,
    const torch::Tensor& conf_,
    const torch::Tensor& pred_cls_,
    const torch::Tensor& target_cls_,
    bool plot /*= false*/,
    const std::string& save_dir /*= "."*/,
    const std::vector<std::string>& names /*= {}*/,
    float eps /*= 1e-16*/
)
{
    auto tp = tp_.cpu().to(torch::kFloat32);
    auto conf = conf_.cpu().to(torch::kFloat32);
    auto pred_cls = pred_cls_.cpu().to(torch::kFloat32);
    auto target_cls = target_cls_.cpu().to(torch::kFloat32);

    //std::cout << " after sort tp: " << tp.sizes() << "conf: " << conf.sizes() << " pred:" << pred_cls.sizes() << " target_cls " << target_cls.sizes() << std::endl; 

    torch::Tensor i = torch::argsort(conf, -1, true).squeeze(-1);
    std::cout << "i " << i.sizes() << std::endl;
    if(i.size(0)!= 0)
    {
        conf = conf.index({ i });
        tp = tp.index({ i });
        pred_cls = pred_cls.index({ i });
    }
    //std::cout << " after sort tp: " << tp.sizes() << "conf: " << conf.sizes() << " pred:" << pred_cls.sizes() << std::endl; 

    // 去重操作
    torch::Tensor unique_classes, tmp_tensor, nt;
    std::tie(unique_classes, tmp_tensor, nt) = torch::_unique2(target_cls, true, false, true);

    int nc = unique_classes.size(0);
    std::cout << "unique_classes: " << unique_classes.sizes() << " ignore: " << tmp_tensor.sizes() << " nt " << nt.sizes() << std::endl;
    torch::Tensor x = torch::linspace(0, 1, 1000);
    std::vector<torch::Tensor> v_prec_values;

    torch::Tensor ap = torch::zeros({ nc, tp.size(1) }, torch::kFloat32);
    torch::Tensor p_curve = torch::zeros({ nc, 1000 }, torch::kFloat32);
    torch::Tensor r_curve = torch::zeros({ nc, 1000 }, torch::kFloat32);
    //std::cout << "ap: " << ap.sizes() << " p_curve: " << p_curve.sizes() << " r_curve: " << r_curve.sizes() << std::endl;
    for (int ci = 0; ci < nc; ++ci)
    {
        int c = unique_classes[ci].item().toInt();
        i = pred_cls == c;
        std::cout << " (2) i"  << i.sizes() << std::endl;
        auto n_l = nt[ci].item<int>();
        auto n_p = i.sum().item<int>();
        std::cout << n_p << " " << n_l << std::endl;
        if (n_p == 0 || n_l == 0)
            continue;

        // Accumulate FPs and TPs
        auto fpc = (1 - tp.index({ i })).cumsum(0);
        auto tpc = tp.index({ i }).cumsum(0);

        // Recall 
        auto recall = tpc / (float(n_l) + eps);
        //  r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
        r_curve[ci] = torch_interp(x.flip(0), conf.index({ i }),
            recall.index({ torch::indexing::Slice(), 0 }), 0.f);

        auto precision = tpc / (tpc + fpc);
        p_curve[ci] = torch_interp(x.flip(0), conf.index({ i }),
            precision.index({ torch::indexing::Slice(), 0 }), 1.f);

        for (int j = 0; j < tp.size(1); j++)
        {
            auto [ap_v, mpre, mrec] = compute_ap(
                recall.index({ torch::indexing::Slice(), j }),
                precision.index({ torch::indexing::Slice(), j })
            );
            if (j == 0)
            {
                auto v_prec_value = torch_interp(x, mrec, mpre);
                v_prec_values.push_back(v_prec_value);
            }
        }
    }
    
    torch::Tensor prec_values = torch::stack(v_prec_values);
    auto f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps);

    i = smooth(f1_curve.mean(0), 0.1f).argmax();    // max F1 index
    auto p = p_curve.index({ torch::indexing::Slice(), i }).clone();
    auto r = r_curve.index({ torch::indexing::Slice(), i }).clone();
    auto f1 = f1_curve.index({ torch::indexing::Slice(), i }).clone();
    tp = (r * nt).round();
    auto fp = (tp / (p + eps) - tp).round();

    return std::make_tuple( tp, fp, p, r, f1, ap, unique_classes.to(torch::kInt), p_curve, r_curve, f1_curve, x, prec_values );
}
