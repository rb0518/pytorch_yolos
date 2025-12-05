#include "ops.h"

torch::Tensor scale_boxes(std::tuple<int, int> img1_shape, 
    torch::Tensor boxes, std::tuple<int, int> img0_shape, 
    std::tuple<std::tuple<float, float>, std::tuple<int, int>> ratio_pad, 
    bool padding, bool xywh)
{
    float gain = 0.0f;
    auto [ratio, pad] = ratio_pad;
    if (std::get<0>(ratio) == 0.0 && std::get<1>(ratio) == 0.0)
    {
        ratio = std::make_tuple(std::get<0>(img1_shape) / std::get<0>(img0_shape),
            std::get<1>(img1_shape) / std::get<1>(img0_shape));
        gain = std::min(std::get<0>(ratio), std::get<1>(ratio));
        pad = std::make_tuple(int((std::get<1>(img1_shape) - std::get<1>(img0_shape) * gain) / 2.0 - 0.1), 
                              int((std::get<0>(img1_shape) - std::get<0>(img0_shape) * gain) / 2.0 - 0.1)
            );
    }
    else
    {
        gain = std::get<0>(ratio);
    }

    if (padding)
    {
        boxes.index({ torch::indexing::Slice(), 0 }).sub_(std::get<0>(pad));
        boxes.index({ torch::indexing::Slice(), 1 }).sub_(std::get<1>(pad));
        if (false == xywh)
        {
            boxes.index({ torch::indexing::Slice(), 2 }).sub_(std::get<0>(pad));
            boxes.index({ torch::indexing::Slice(), 3 }).sub_(std::get<1>(pad));
        }
    }
    boxes.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 4) }).div_(gain);

    return clip_boxes(boxes, img0_shape);
}

torch::Tensor clip_boxes(torch::Tensor boxes, std::tuple<int, int> shape)
{
    auto [shape0, shape1] = shape;
    boxes.index({ "...", 0 }).clamp_(0, shape1);
    boxes.index({ "...", 1 }).clamp_(0, shape0);
    boxes.index({ "...", 2 }).clamp_(0, shape1);
    boxes.index({ "...", 3 }).clamp_(0, shape0);

    return boxes;
}

torch::Tensor xyxy2xywh(const torch::Tensor& x)
{
    // 检查输入张量的最后一个维度是否为4
    TORCH_CHECK(x.size(-1) == 4, "input shape last dimension expected 4 but input shape is ", x.sizes());
    // 创建与输入相同类型的输出张量
    auto y = torch::empty_like(x);

    y.index_put_({"...", 0}, (x.index({"...", 0}) + x.index({"...", 2})) / 2);
    y.index_put_({"...", 1}, (x.index({"...", 1}) + x.index({"...", 3})) / 2);
    y.index_put_({"...", 2}, x.index({"...", 2}) - x.index({"...", 0}));
    y.index_put_({"...", 3}, x.index({"...", 3}) - x.index({"...", 1}));
    return y;
}

torch::Tensor xywh2xyxy(const torch::Tensor& x) 
{
    TORCH_CHECK(x.size(-1) == 4, "input shape last dimension expected 4 but input shape is ", x.sizes());
    
    auto y = torch::empty_like(x);
   
    // 计算左上角坐标
    y.index_put_({"...", 0}, x.index({"...", 0}) - x.index({"...", 2}) / 2);
    y.index_put_({"...", 1}, x.index({"...", 1}) - x.index({"...", 3}) / 2);
    // 计算右下角坐标
    y.index_put_({"...", 2}, x.index({"...", 0}) + x.index({"...", 2}) / 2);
    y.index_put_({"...", 3}, x.index({"...", 1}) + x.index({"...", 3}) / 2);
    return y;
}

torch::Tensor xywhn2xyxy(const torch::Tensor& x, int w /*= 640*/, int h /*= 640*/, int padw /*= 0*/, int padh /*= 0*/) 
{
    TORCH_CHECK(x.size(-1) == 4, "input shape last dimension expected 4 but input shape is ", x.sizes());
    auto y = torch::empty_like(x);
    
    // 计算转换后的坐标
    y.index_put_({"...", 0}, w * (x.index({"...", 0}) - x.index({"...", 2}) / 2) + padw);  // top left x
    y.index_put_({"...", 1}, h * (x.index({"...", 1}) - x.index({"...", 3}) / 2) + padh);  // top left y
    y.index_put_({"...", 2}, w * (x.index({"...", 0}) + x.index({"...", 2}) / 2) + padw);  // bottom right x
    y.index_put_({"...", 3}, h * (x.index({"...", 1}) + x.index({"...", 3}) / 2) + padh);  // bottom right y
    
    return y;
}

torch::Tensor xyxy2xywhn(const torch::Tensor& x, int w /*= 640*/, int h /*= 640*/, bool clip /*= false*/, float eps /*= 0.0*/) {
    // 检查输入张量的最后一个维度是否为4
    TORCH_CHECK(x.size(-1) == 4, "input shape last dimension expected 4 but input shape is ", x.sizes());
    
    // 创建与输入相同类型的输出张量
    auto y = torch::empty_like(x);
    
    // 如果需要裁剪到图像边界，前期简化，暂时不用，未完善
    if (clip) {
//        auto clipped_x = clip_boxes(x, {h - eps, w - eps});
//        x = clipped_x;
    }
    
    // 计算归一化中心坐标和宽高
    y.index_put_({"...", 0}, ((x.index({"...", 0}) + x.index({"...", 2})) / 2) / w);  // x center
    y.index_put_({"...", 1}, ((x.index({"...", 1}) + x.index({"...", 3})) / 2) / h);  // y center
    y.index_put_({"...", 2}, (x.index({"...", 2}) - x.index({"...", 0})) / w);        // width
    y.index_put_({"...", 3}, (x.index({"...", 3}) - x.index({"...", 1})) / h);        // height
    
    return y;
}

torch::Tensor xyn2xy(const torch::Tensor& x, int w, int h, int padw, int padh)
{
    TORCH_CHECK(x.size(-1) == 2, "input shape last dimension expected 2 but input shape is ", x.sizes());
    auto y = torch::empty_like(x);
    
    // 计算转换后的坐标
    y.index_put_({"...", 0}, w * x.index({"...", 0}) + padw);  // top left x
    y.index_put_({"...", 1}, h * x.index({"...", 1}) + padh);  // top left y
   
    return y;
}
