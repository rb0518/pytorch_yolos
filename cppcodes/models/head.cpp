#include "head.h"

#include <string>
#include <vector>
#include <algorithm>    // std::min

#include "tal.h"

/*
    解决libtorch中Sequential里须明确forward方法的限制
    注意：在此Sequential下所有子类的forward都与下现的forward定义要保持一致
*/
struct StackSequentailImpl : torch::nn::SequentialImpl {
  using torch::nn::SequentialImpl::SequentialImpl;

  torch::Tensor forward(torch::Tensor x) {
    return SequentialImpl::forward(x);
  }
};

TORCH_MODULE(StackSequentail);

DetectImpl::DetectImpl(int _nc /*= 80*/, std::vector<int> _ch /*= {}*/, bool _legacy /*= false*/)
{
    nc = _nc;               // number of classes
    nl = _ch.size();        // number of detection layers
    reg_max = 16;           // DFL channels(ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
    no = nc + reg_max * 4;  // number of ouputs per anchors
    stride = torch::zeros({nl});
    legacy = _legacy;

    //std::cout << "ch: [ " << _ch[0] <<" " << _ch[1] << " " << _ch[2] << " ]\n";
    c2 = std::max(16, std::max(_ch[0] / 4, reg_max * 4));
    c3 = std::max(_ch[0], std::min(nc, 100));

    //std::cout << "c2: " << c2 << " c3: " << c3 << std::endl;
    for(int i = 0; i < _ch.size(); i++)
    {
        auto x = _ch[i];
        //std::cout << " x : _ch: " << x << " c2 " << c2 << " c3: " << c3 << std::endl;
        //cv2->push_back(Detect_2Conv_Conv2d(x, c2, 4* reg_max));
        cv2->push_back(torch::nn::Sequential(
            Conv(x, c2, 3), Conv(c2, c2, 3), 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(c2, 4 * reg_max, 1))
            ));
      
        if(legacy)
        {
            cv3->push_back(torch::nn::Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), 
                torch::nn::Conv2d(torch::nn::Conv2dOptions(c3, nc, 1))
            ));
        }
        else{
            cv3->push_back(torch::nn::Sequential(
                StackSequentail(DWConv(x, x, 3), Conv(x, c3, 1)),
                StackSequentail(DWConv(c3, c3, 3), Conv(c3,c3, 1)),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(c3, nc, 1))
            ));
        }       
    }

    if(reg_max > 1)
        dfl = std::make_shared<DFLImpl>(reg_max);
    else
        dfl = std::make_shared<torch::nn::IdentityImpl>();
    
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    register_module("dfl", dfl.ptr());

    if(end2end)
    {   // 暂时未用到，后续完成测试
        // libtorch 没有copy.deepcopy 
        for(auto x : _ch)
        {
            one2one_cv2->push_back(torch::nn::Sequential(
                        Conv(x, c2, 3), Conv(c2, c2, 3), 
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(c2, 4 * reg_max, 1))
                    ));
            if(legacy)
            {
                one2one_cv3->push_back(torch::nn::Sequential(
                        Conv(x, c3, 3), Conv(c3, c3, 3), 
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(c3, nc, 1))
                    ));
            }
            else{
                one2one_cv3->push_back(torch::nn::Sequential(
                        DWConv(x, x, 3), Conv(x, c3, 1),
                        DWConv(c3, c3, 3), Conv(c3,c3, 1),
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(c3, nc, 1))
                    ));
            }
        }
        register_module("one2one_cv2", one2one_cv2);
        register_module("one2one_cv3", one2one_cv3);
    }
    std::cout << "DetectImpl init ok \n";
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>>
        DetectImpl::forward(std::vector<torch::Tensor> x)
{
    // C++函数不能简单各分支返回不一样的内容
    std::vector<torch::Tensor> x_ret;
    if(end2end){

    }
    
    for(int i = 0; i < nl; i++)
    {
        auto cv2_any = cv2[i]->as<torch::nn::Sequential>();
        auto cv3_any = cv3[i]->as<torch::nn::Sequential>();
        x_ret.emplace_back(torch::cat({cv2_any->forward(x[i]), cv3_any->forward(x[i])}, 1));
    }

    if (is_training())   // 如果是训练 
        return std::make_tuple(torch::empty({ 1 }), x_ret);

    auto y = _inference(x_ret);

    return { y, x_ret };
}

std::tuple<torch::Tensor, std::unordered_map<std::string, std::vector<torch::Tensor>>>
DetectImpl::forward_end2end(std::vector<torch::Tensor> x) 
{
    std::vector<torch::Tensor> x_detach;
    for (const auto& xi : x) {
        x_detach.emplace_back(xi.detach());
    }

    std::vector<torch::Tensor> one2one;
    for (int i = 0; i < this->nl; ++i) 
    {
        auto o2o_cv2 = one2one_cv2[i]->as<torch::nn::AnyModule>();
        auto o2o_cv3 = one2one_cv3[i]->as<torch::nn::AnyModule>();

        auto cv2_output = o2o_cv2->forward(x_detach[i]);
        auto cv3_output = o2o_cv3->forward(x_detach[i]);

        one2one.emplace_back(torch::cat({ cv2_output, cv3_output }, 1));
    }

    // 处理输入张量x
    for (int i = 0; i < x.size(); ++i) 
    {
        auto cv2_any = cv2[i]->as<torch::nn::AnyModule>();
        auto cv3_any = cv3[i]->as<torch::nn::AnyModule>();

        auto cv2_output = cv2_any->forward(x[i]);
        auto cv3_output = cv3_any->forward(x[i]);
        x[i] = torch::cat({ cv2_output, cv3_output }, 1);
    }

    if (this->is_training()) 
    {
        return std::make_tuple(
            torch::Tensor(),
            std::unordered_map<std::string, std::vector<torch::Tensor>>{
                {"one2many", x},
                { "one2one", one2one }
                });
    }

    auto y = this->_inference(one2one);
    y = postprocess(y.permute({ 0, 2, 1 }), this->max_det, this->nc);

    // 根据export标志返回结果
    if (this->_export_) {
        return std::make_tuple(y, std::unordered_map<std::string, std::vector<torch::Tensor>>());
    }
        
    return std::make_tuple(
            y,
            std::unordered_map<std::string, std::vector<torch::Tensor>>{
                {"one2many", x},
                { "one2one", one2one }
            });
}

/*
    """
    Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.
    Args:
        x (List[torch.Tensor]): List of feature maps from different detection layers.
    Returns:
        (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
    """
*/
at::Tensor DetectImpl::_inference(std::vector<at::Tensor> x)
{
    auto shape = x[0].sizes();  
    //std::cout << shape << std::endl;
    std::vector<at::Tensor> tmp;
    for (int i = 0; i < x.size(); i++)
        //x[i] = x[i].view({ shape[0], no, -1 });
        tmp.emplace_back(x[i].clone().view({ shape[0], no, -1 }));
    auto x_cat = torch::cat(tmp, 2);
    //auto x_cat = torch::cat(x, 2);
    //std::cout << "x_cat: " << x_cat.sizes() << std::endl;

    if (_dynamic_ || shape != shape_)
    {
        //std::cout << "make_anchors start: " << this->stride.sizes() << " " << this->stride <<"\n";
        auto [ma_anchors, ma_strides] = make_anchors(x, this->stride, 0.5f);
        //std::cout << "make_anchors ma_anchors " <<ma_anchors.sizes() << " ma_strides: " << ma_strides.sizes()<< std::endl ;
        
        this->anchors = ma_anchors.transpose(0, 1);   // [N, 2] ==> [2, N]
        this->strides = ma_strides.transpose(0, 1);

        shape_ = shape;
    }
    at::Tensor box;
    at::Tensor cls;
    at::Tensor dbox;
    if (_export_)
    {
        box = x_cat.index({ "...", torch::indexing::Slice(0, reg_max * 4)});
        cls = x_cat.index({ "...", torch::indexing::Slice(reg_max * 4) });
    }
    else
    {
        auto x_cat_split = x_cat.split({ reg_max * 4, nc }, 1);
        box = x_cat_split[0];
        cls = x_cat_split[1];
        std::cout << "box: " << box.sizes() << std::endl;
        std::cout << "cls: " << cls.sizes() << std::endl;
    }
    if (_export_)
    {
        // do nothing
    }
    else
    {
        dbox = decode_bboxes(dfl.forward(box), anchors.unsqueeze(0)) * strides;
    }
    return torch::cat({dbox, cls.sigmoid()}, 1);
}

at::Tensor DetectImpl::decode_bboxes(torch::Tensor bboxes, torch::Tensor anchors, bool xywh /*= true*/)
{
    bool is_xywh = xywh & !end2end & !xyxy;
    return dist2bbox(bboxes, anchors, is_xywh , 1);
}


void DetectImpl::bias_init()
{
    torch::NoGradGuard nograd;
    std::cout << "bias_init: \n";

    for(int i = 0; i < nl; i++)
    {
        double s = this->stride[i].item().toDouble();
        double log_val = std::log(5.0 / nc / std::pow(640.0 / s, 2));

        auto cv2_any = cv2[i]->as<torch::nn::Sequential>();
        std::cout << i << " cv2 size: " << cv2_any->size() << " childrens: " << cv2_any->children().size() << std::endl;
        auto cv2_module = cv2_any->children().back()->as<torch::nn::Conv2d>();
        if (cv2_module != nullptr)
        {
            if (cv2_module->bias.defined())
                cv2_module->bias.data().fill_(1.0);
            else
                std::cout << "cv2->bias not defined." << std::endl;
        }

        auto cv3_any = cv3[i]->as<torch::nn::Sequential>();
        std::cout << i << " cv3 size: " << cv3_any->size() << " childrens: " << cv3_any->children().size() << std::endl;
        auto cv3_module = cv3_any->children().back()->as<torch::nn::Conv2d>();
        if (cv3_module != nullptr)
        {
            if (cv3_module->bias.defined())
                cv3_module->bias.data().index_put_(
                    { torch::indexing::Slice(0, nc) }, log_val);
            else
                std::cout << "cv2->bias not defined." << std::endl;
        }
    }
}

torch::Tensor DetectImpl::postprocess(torch::Tensor preds, int max_det, int nc /*= 80*/) 
{
    auto batch_size = preds.size(0);
    int anchors = int(preds.size(1));
    auto dtype = preds.dtype();
    auto device = preds.device();

    // 分割boxes和scores
    auto preds_split = preds.split({ 4, nc }, -1);
    auto boxes = preds_split[0];
    auto scores = preds_split[1];

    // 获取每行最大值的索引
    auto [scores_max,indices] = scores.amax(-1).topk(std::min(max_det, anchors));
    indices = indices.unsqueeze(-1);
    boxes = boxes.gather(1, indices.repeat({ 1, 1, 4 }));
    scores = scores.gather(1, indices.repeat({ 1, 1, nc }));
    std::tie(scores, indices) = scores.flatten(1).topk(std::min(max_det, anchors));

    auto final_indices = torch::remainder(indices.clone(),nc).unsqueeze(-1).to(torch::kFloat);
    scores = scores.unsqueeze(-1);

    auto i = torch::arange(batch_size, torch::TensorOptions().dtype(dtype).device(device)).unsqueeze(-1);
    auto anchor_indices = indices.div(nc).to(torch::kLong);
    auto select_boxes = boxes.index({ i, anchor_indices });
    return torch::cat({ select_boxes,scores, final_indices }, -1);
}