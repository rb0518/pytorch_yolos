#include "head.h"

#include <string>
#include <vector>
#include <algorithm>    // std::min

#include "tal.h"

class Detect_2Conv_Conv2d : public torch::nn::Module
{
public:
    Detect_2Conv_Conv2d(int x, int c2, int reg_or_nc)
    {
        cv1 = Conv(x, c2, 3);
        cv2 = Conv(c2, c2, 3);
        c = torch::nn::Conv2d(torch::nn::Conv2dOptions(c2, reg_or_nc, 1));
        register_module("0", cv1);
        register_module("1", cv2);
        register_module("2", c);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return c->forward(cv2->forward(cv1->forward(x)));
    }

    void init_bias(bool b_cv2, int nc, double d_f)
    {
        std::cout << "Detect_2Conv_Conv2d::init_bias ";
        if (this->c->bias.defined())
        {
            if(b_cv2)
            {
                std::cout << "cv2: fill with 1.0\n";
                c->bias.data().fill_(1.0);
            }
            else{
                std::cout << "cv3: fill with: " << nc << " " << d_f << std::endl;
                c->bias.data().index_put_({ torch::indexing::Slice(0, nc) }, d_f);
            }
        }
        else{
            LOG(ERROR) << "Conv2d not define bias.";
        }
        std::cout << " over\n";
    }

public:
    Conv cv1;
    Conv cv2;
    torch::nn::Conv2d c{nullptr};
};

class Detect_cv3_2DWConvConv_Conv2d : public torch::nn::Module
{
public:
    Detect_cv3_2DWConvConv_Conv2d(int x, int c3, int nc)
    {
        dw1 = DWConv(x, x, 3);
        cv1 = Conv(x, c3, 1);
        dw2 = DWConv(c3, c3, 3);
        cv2 = Conv(c3, c3, 1);
        c = torch::nn::Conv2d(torch::nn::Conv2dOptions(c3, nc, 1));

        register_module("0-0", dw1);
        register_module("0-1", cv1);

        register_module("1-0", dw2);
        register_module("1-1", cv2);

        register_module("2", c);
    }
    torch::Tensor forward(torch::Tensor x)
    {
        return c->forward(cv2->forward(dw2->forward(cv1->forward(dw1->forward(x)))));
    }

    void init_bias(bool b_cv2, int nc, double d_f)  // 保持与cv2函数一致，但不管b_cv2的值了
    {
        std::cout << "Detect_cv3_2DWConvConv_Conv2d::init_bias ";
        if (this->c->bias.defined())
        {
            std::cout << "cv3: fill with: " << nc << " " << d_f << std::endl;            
            c->bias.data().index_put_({ torch::indexing::Slice(0, nc) }, d_f);
        }
        else{
            LOG(ERROR) << "Conv2d not define bias.";
        }
        std::cout << " over. \n";
    }
public:
    DWConv  dw1;
    Conv    cv1;
    DWConv  dw2;
    Conv    cv2;
    torch::nn::Conv2d c{nullptr};
};


DetectImpl::DetectImpl(int _nc /*= 80*/, std::vector<int> _ch /*= {}*/)
{
    nc = _nc;               // number of classes
    nl = _ch.size();        // number of detection layers
    reg_max = 16;           // DFL channels(ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
    no = nc + reg_max * 4;  // number of ouputs per anchors
    stride = torch::zeros({nl});

    //std::cout << "ch: [ " << _ch[0] <<" " << _ch[1] << " " << _ch[2] << " ]\n";
    c2 = std::max(16, std::max(_ch[0] / 4, reg_max * 4));
    c3 = std::max(_ch[0], std::min(nc, 100));

    //std::cout << "c2: " << c2 << " c3: " << c3 << std::endl;
    for(int i = 0; i < _ch.size(); i++)
    {
        auto x = _ch[i];
        //std::cout << " x : _ch: " << x << " c2 " << c2 << " c3: " << c3 << std::endl;
        cv2->push_back(Detect_2Conv_Conv2d(x, c2, 4* reg_max));

        if(legacy)
        {
            cv3->push_back(Detect_2Conv_Conv2d(x, c3, nc));
        }
        else{
            cv3->push_back(Detect_cv3_2DWConvConv_Conv2d(x, c3, nc));
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
            one2one_cv2->push_back(Detect_2Conv_Conv2d(x, c2, 4*reg_max));
            if(legacy)
            {
                one2one_cv3->push_back(Detect_2Conv_Conv2d(x, c3, nc));
            }
            else{
                one2one_cv3->push_back(Detect_cv3_2DWConvConv_Conv2d(x, c3, nc));
            }
        }
        register_module("one2one_cv2", one2one_cv2);
        register_module("one2one_cv3", one2one_cv3);
    }
    //std::cout << "DetectImpl init ok \n";
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>>
        DetectImpl::forward(std::vector<torch::Tensor> x)
{
    // C++函数不能简单各分支返回不一样的内容
    if(end2end){

    }
    
    for(int i = 0; i < nl; i++)
    {
        // std::cout << "Detect i: " << i << " input: " << x[i].sizes() << "\n";
        auto cv2_any = cv2[i]->as<Detect_2Conv_Conv2d>();
        if(legacy)
        {
            auto cv3_any = cv3[i]->as<Detect_2Conv_Conv2d>();
            x[i] = torch::cat({ cv2_any->forward(x[i]), cv3_any->forward(x[i]) }, 1);
        }
        else{
            auto cv3_any = cv3[i]->as<Detect_cv3_2DWConvConv_Conv2d>();
            x[i] = torch::cat({ cv2_any->forward(x[i]), cv3_any->forward(x[i]) }, 1);
        }
        // std::cout << "output: " << x[i].sizes() <<"\n";
    }

    if (is_training())   // 如果是训练 
        return std::make_tuple(torch::empty({ 1 }), x);

    auto y = _inference(x);

    return { y, x };
}

std::tuple<torch::Tensor, std::unordered_map<std::string, std::vector<torch::Tensor>>>
DetectImpl::forward_end2end(std::vector<torch::Tensor>& x) 
{
    std::vector<torch::Tensor> x_detach;
    for (const auto& xi : x) {
        x_detach.push_back(xi.detach());
    }

    std::vector<torch::Tensor> one2one;
    for (int i = 0; i < this->nl; ++i) 
    {
        auto o2o_cv2 = one2one_cv2[i]->as<torch::nn::AnyModule>();
        auto o2o_cv3 = one2one_cv3[i]->as<torch::nn::AnyModule>();

        auto cv2_output = o2o_cv2->forward(x_detach[i]);
        auto cv3_output = o2o_cv3->forward(x_detach[i]);

        one2one.push_back(torch::cat({ cv2_output, cv3_output }, 1));
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
    std::cout << "bias_init: \n";
    for(int i = 0; i < nl; i++)
    {
        auto cv2_any = cv2[i]->as<Detect_2Conv_Conv2d>();
        cv2_any->init_bias(true, nc, 1.f);

        double s = this->stride[i].item().toDouble();
        double log_val = std::log(5.0 / nc / std::pow(640.0 / s, 2));
        //std::cout << "s: " << s << " nc: " << nc << " fill_val: " << log_val << std::endl;
        printf("s: %d nc: %d fill_val: %lf", int(s), int(nc), log_val);
        if (legacy)
        {
            auto cv3_any = cv3[i]->as<Detect_2Conv_Conv2d>();
            cv3_any->init_bias(false, nc, log_val);
        }
        else {
            auto cv3_any = cv3[i]->as<Detect_cv3_2DWConvConv_Conv2d>();
            cv3_any->init_bias(false, nc, log_val);
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

    auto final_indices = (indices.clone() % nc).unsqueeze(-1).to(torch::kFloat);
    scores = scores.unsqueeze(-1);

    auto i = torch::arange(batch_size, torch::TensorOptions().dtype(dtype).device(device)).unsqueeze(-1);
    auto anchor_indices = indices.div(nc).to(torch::kLong);
    auto select_boxes = boxes.index({ i, anchor_indices });
    return torch::cat({ select_boxes,scores, final_indices }, -1);
}