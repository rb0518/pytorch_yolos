#include "head.h"

#include <string>
#include <vector>
#include <algorithm>    // std::min
#include <cmath>
#include "tal.h"
#include "globaldef.h"
#include "BaseData.h"

#include "ops.h"

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

// ========================= RT-DETR ==========================

RTDETRDecoderImpl::RTDETRDecoderImpl(int nc, std::vector<int> inchs,
    int hd, int nq, int ndp, int nh, int ndl, int d_ffn, 
    float dropout, int eval_idx, int nd, 
    float label_noise_ration, float box_noise_scale, bool learnt_init_query)
{
    m_nc = nc;
    m_nl = inchs.size();
    m_hidden_dim = hd;
    m_nhead = nh;
    m_num_queries = nq;
    m_num_decoder_layers = ndl;

    // Backbone feature projection
    for (auto x : inchs)
    {
        m_input_proj->push_back(torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(x, hd, 1).bias(false)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(hd))
        ));
    }
    register_module("input_proj", m_input_proj);

    // Transformer module
    m_decoder = DeformableTransformerDecoder(hd, ndl, eval_idx);
    m_decoder->createLayers(hd, nh, d_ffn, dropout, m_nl, ndp);
    register_module("decoder", m_decoder);

    // Denoising part
    m_denoising_class_embed = torch::nn::Embedding(torch::nn::EmbeddingOptions(nc, hd));
    register_module("denoising_class_embed", m_denoising_class_embed);
    m_num_denoising = nd;
    m_label_noise_ratio = label_noise_ration;
    m_box_noise_scale = box_noise_scale;


    // Decoder embedding
    m_learnt_init_query = learnt_init_query;
    if (learnt_init_query)
    {
        m_tgt_embed = torch::nn::Embedding(torch::nn::EmbeddingOptions(nq, hd));
        register_module("tgt_embed", m_tgt_embed);
    }
    m_query_pos_head = MLP(hd, hd, 4, 3);
    register_module("query_pos_head", m_query_pos_head);

    // Encoder head
    m_enc_output = torch::nn::Sequential(
        torch::nn::Linear(hd, hd),
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({ int64_t(hd) }))
    );
    m_enc_score_head = torch::nn::Linear(hd, nc);
    m_enc_bbox_head = MLP(hd, hd, 4, 3);
    register_module("enc_output", m_enc_output);
    register_module("enc_score_head", m_enc_score_head);
    register_module("enc_bbox_head", m_enc_bbox_head);

    // Decoder head
    for (int i = 0; i < ndl; i++)
    {
        m_dec_score_head->push_back(torch::nn::Linear(hd, nc));
        m_dec_bbox_head->push_back(std::make_shared<MLPImpl>(hd, hd, 4, 3));
    }
    register_module("dec_score_head", m_dec_score_head);
    register_module("dec_bbox_head", m_dec_bbox_head);

    _reset_parameters();
}

std::tuple<std::optional<torch::Tensor>, 
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>,
    std::optional<VariantData>>
    get_cdn_group(VariantData batch, int num_classes, int num_queries,
        torch::Tensor class_embed, int num_dn = 100,
        float cls_noise_ratio = 0.5f,
        float box_noise_scale = 0.5f,
        bool training = false)
{
    if (training == false || num_dn <= 0 || batch.size() == 0)
        return { std::nullopt, std::nullopt, std::nullopt, std::nullopt };

    std::vector<int> gt_groups = std::get<std::vector<int>>(batch["gt_groups"]);
    auto total_num = std::accumulate(gt_groups.begin(), gt_groups.end(), 0);
    int max_nums = *std::max_element(gt_groups.begin(), gt_groups.end());
    if (max_nums == 0)
        return { std::nullopt, std::nullopt, std::nullopt, std::nullopt };

    int num_group = std::max(1, num_dn / max_nums);
    int bs = gt_groups.size();
    auto gt_cls = std::get<torch::Tensor>(batch["cls"]); //[bs*num, ]
    auto gt_bbox = std::get<torch::Tensor>(batch["bboxes"]);    //[bs*num, 4]
    auto b_idx = std::get<torch::Tensor>(batch["batch_idx"]);

    auto dn_cls = gt_cls.repeat({ 2 * num_group }); // (2*num_groups*bs*num, )
    auto dn_bbox = gt_bbox.repeat({ 2 * num_group, 1 });
    auto dn_b_idx = b_idx.repeat({ 2 * num_group }).view(-1);
    std::cout << "dn_bbox: " << dn_bbox.sizes() << " gt_bbox: " << gt_bbox.sizes() << std::endl;

    auto neg_idx = torch::arange(total_num * num_group, torch::TensorOptions(torch::kLong).device(gt_bbox.device())) + num_group * total_num;

    if (cls_noise_ratio > 0.f)
    {
        auto mask = torch::rand(dn_cls.size(0)) < (cls_noise_ratio * 0.5);
        auto idx = torch::nonzero(mask).squeeze(-1);
        auto new_label = torch::randint_like(idx, 0, num_classes, torch::TensorOptions(dn_cls.dtype()).device(dn_cls.device()));
        dn_cls.index({ idx }) = new_label;
    }

    if (box_noise_scale > 0.f)
    {
        auto known_bbox = xywh2xyxy(dn_bbox);
        auto diff = (dn_bbox.index({ "...", torch::indexing::Slice(2, torch::indexing::None) }) * 0.5).repeat({1, 2}) * box_noise_scale;

        auto rand_sign = torch::randint_like(dn_bbox, 0, 2) * 2.0 - 1.0;
        auto rand_part = torch::rand(dn_bbox.sizes(), torch::TensorOptions(dn_bbox.dtype()).device(dn_bbox.device()).layout(dn_bbox.layout()));

        rand_part.index({ neg_idx }).add_(1.0);
        rand_part.mul_(rand_sign);
        known_bbox = known_bbox + rand_part * diff;
        known_bbox.clip_(0.0, 1.0);
        dn_bbox = xyxy2xywh(known_bbox);
        dn_bbox = torch::logit(dn_bbox, 1e-6);
    }
    
    num_dn = int(max_nums * 2 * num_group);
    auto dn_cls_embed = class_embed.index({ dn_cls });
    auto padding_cls = torch::zeros({ bs, num_dn, dn_cls_embed.sizes().back() }, torch::TensorOptions(gt_cls.device()));
    auto padding_bbox = torch::zeros({ bs, num_dn, 4 }, torch::TensorOptions(gt_bbox.device()));
    std::vector<torch::Tensor> tmp_ranges;
    for (auto num : gt_groups)
    {
        tmp_ranges.emplace_back(torch::range(0, num, torch::kLong));
    }
    auto map_indices = torch::cat(tmp_ranges);
    std::vector<torch::Tensor> tmp_pos_idx;
    for (int i = 0; i < num_group; i++)
        tmp_pos_idx.push_back(map_indices + max_nums * i);
    auto pos_idx = torch::stack(tmp_pos_idx);
    std::vector<torch::Tensor> tmp_map_indices;
    for (int i = 0; i < 2 * num_group; i++)
        tmp_pos_idx.push_back(map_indices + max_nums * i);
    map_indices = torch::cat(tmp_pos_idx);
    padding_cls.index_put_({ dn_b_idx, map_indices }, dn_cls_embed);
    padding_bbox.index_put_({ dn_b_idx, map_indices }, dn_bbox);

    int tgt_size = num_dn + num_queries;
    auto attn_mask = torch::zeros({ tgt_size, tgt_size }, torch::kBool);
    attn_mask.index_put_({ torch::indexing::Slice(num_dn, torch::indexing::None),
        torch::indexing::Slice(torch::indexing::None, num_dn) }, true);

    for (int i = 0; i < num_group; i++) {
        if (i == 0)
            attn_mask.index_put_({ torch::indexing::Slice(max_nums * 2 * i, max_nums * 2 * (i + 1)),
                torch::indexing::Slice(max_nums * 2 * (i + 1), num_dn) }, true);
        else if (i == (num_group - 1))
            attn_mask.index_put_({ torch::indexing::Slice(max_nums * 2 * i, max_nums * 2 * (i + 1)),
                torch::indexing::Slice(0, max_nums * i * 2) }, true);
        else
        {
            attn_mask.index_put_({ torch::indexing::Slice(max_nums * 2 * i, max_nums * 2 * (i + 1)),
                torch::indexing::Slice(max_nums * 2 * (i + 1), num_dn) }, true);
            attn_mask.index_put_({ torch::indexing::Slice(max_nums * 2 * i, max_nums * 2 * (i + 1)),
                torch::indexing::Slice(0, max_nums * 2 * 2) }, true);
        }
    }
    VariantData dn_meta;
    std::vector<int64_t> tmp_gt_groups;
    for (int i = 0; i < gt_groups.size(); i++)
        tmp_gt_groups.push_back(int64_t(gt_groups[i]));
    auto tmp_p = pos_idx.cpu().split(tmp_gt_groups, 1);
    std::vector<torch::Tensor> dn_pos_idx;
    for (auto p : tmp_p)
        dn_pos_idx.push_back(p.reshape(-1));

    dn_meta["dn_pos_idx"] = dn_pos_idx;
    dn_meta["dn_num_group"] = num_group;
    dn_meta["dn_num_split"] = std::vector<int>({ num_dn, num_queries });

    return { padding_cls.to(class_embed.device()),
        padding_bbox.to(class_embed.device()),
        attn_mask.to(class_embed.device()),
        dn_meta };
}

std::tuple < std::optional<torch::Tensor>,
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<VariantData>> >
    RTDETRDecoderImpl::forward(std::vector<torch::Tensor> x, VariantData batch)
{
    auto [feats, shapes] = _get_encoder_input(x);
    auto [dn_embed, dn_bbox, attn_mask, dn_meta] = get_cdn_group(
        batch,
        m_nc,
        m_num_queries,
        m_denoising_class_embed->weight,
        m_num_denoising,
        m_label_noise_ratio,
        m_box_noise_scale,
        is_training()
        );

    auto [embed, refer_bbox, enc_bboxes, enc_scores] = _get_decoder_input(feats, shapes, dn_embed, dn_bbox);

    // Decoder
    auto [dec_bboxes, dec_scores] = m_decoder->forward(embed,
        refer_bbox, feats, shapes, 
        m_dec_bbox_head, m_dec_score_head, m_query_pos_head, attn_mask);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<VariantData>> x_ret =
        std::make_tuple(dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta);

    std::optional<torch::Tensor> y = std::nullopt;
    if (is_training() == false)
        y = torch::cat({ dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid() }, -1);

    return { y, x_ret };
}

void RTDETRDecoderImpl::_reset_parameters()
{
    auto bias_init_with_prob = [](double prior_prob) {
        return float(std::log((1 - prior_prob) / prior_prob));
        };
    auto bias_cls = bias_init_with_prob(0.01) / 80 * m_nc;

    torch::nn::init::constant_(m_enc_score_head->bias, bias_cls);
    m_enc_bbox_head->init_weights();

    for (int i = 0; i < m_dec_score_head->size(); i++)
    {
        auto cls_ = m_dec_score_head[i]->as<torch::nn::Linear>();
        if(cls_)
            torch::nn::init::constant_(cls_->bias, bias_cls);

        auto reg_ = m_dec_bbox_head[i]->as<MLP>();
        if (reg_)
            reg_->init_weights();
    }
    /*  python linear_init
        bound = 1 / math.sqrt(module.weight.shape[0])
        uniform_(module.weight, -bound, bound)
        if hasattr(module, "bias") and module.bias is not None:
            uniform_(module.bias, -bound, bound)
    */

    int64_t tmp_size = (m_enc_output->children()[0]->as<torch::nn::Linear>())->weight.size(0);
    double bound = 1.0 / std::sqrt(double(tmp_size));
    torch::nn::init::uniform_((m_enc_output->children()[0]->as<torch::nn::Linear>())->weight, -bound, bound);
    torch::nn::init::uniform_((m_enc_output->children()[0]->as<torch::nn::Linear>())->bias, -bound, bound);
   
    torch::nn::init::xavier_uniform_((m_enc_output->children()[0]->as<torch::nn::Linear>())->weight);
    if (m_learnt_init_query)
        torch::nn::init::xavier_uniform_(m_tgt_embed->weight);

    m_query_pos_head->init_xavier_uniform(0);
    m_query_pos_head->init_xavier_uniform(1);
    
    for (int i = 0; i < m_input_proj->size(); i++)
    {
        auto layer_sq = m_input_proj[i]->as<torch::nn::Sequential>();
        auto cv_module = (layer_sq->children())[0]->as<torch::nn::Conv2d>();
        if (cv_module != nullptr)
            torch::nn::init::xavier_uniform_(cv_module->weight);
    }
}

std::tuple < torch::Tensor, std::vector<std::vector<int>> > RTDETRDecoderImpl::_get_encoder_input(std::vector<torch::Tensor> x_)
{
    std::vector<torch::Tensor> x;
    for (int i = 0; i < x_.size(); i++)
    {
        auto feat = x_[i];
        x.push_back(m_input_proj[i]->as<torch::nn::Sequential>()->forward(feat));
    }
    std::vector<torch::Tensor> feats;
    std::vector<std::vector<int>> shapes;
    for (auto feat : x)
    {
        int h = feat.size(2);
        int w = feat.size(3);
        // [b, c, h, w] ==> [b, h*w, c]
        feats.push_back(feat.flatten(2).permute({ 0, 2, 1 }));
        std::cout << " feats: " << feats.back().sizes() << std::endl;
        shapes.emplace_back(std::vector<int>({ h, w }));
    }
    return { torch::cat(feats, 1), shapes };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RTDETRDecoderImpl::_get_decoder_input(torch::Tensor feats,
    std::vector<std::vector<int>> shapes, std::optional<torch::Tensor> dn_embed, std::optional<torch::Tensor> dn_bbox)
{
    int bs = feats.size(0);

    auto [anchors, valid_mask] = _generate_anchors(shapes, feats);
    m_shapes = shapes;

    auto features = m_enc_output->forward(valid_mask * feats);
    auto enc_output_scores = m_enc_score_head->forward(features);

    auto topk_ind = std::get<1>(torch::topk(std::get<0>(enc_output_scores.max(-1)), m_num_queries, 1)).view(-1);
    auto batch_ind = torch::arange(bs, topk_ind.dtype()).unsqueeze(-1).repeat({ 1, m_num_queries }).view(-1);
    auto top_k_features = features.index({ batch_ind, topk_ind }).view({ bs, m_num_queries, -1 });
    auto top_k_anchors = anchors.index({ torch::indexing::Slice(), topk_ind }).view({ bs, m_num_queries, -1 });

    auto refer_bbox = m_enc_bbox_head->forward(top_k_features) + top_k_anchors;

    auto enc_bboxes = refer_bbox.sigmoid();
    if (dn_bbox.has_value())
        refer_bbox = torch::cat({ batch_ind, topk_ind }, 1);
    auto enc_scores = enc_output_scores.index({ batch_ind, topk_ind }).view({ bs, m_num_queries, -1 });

    auto embeddings = m_learnt_init_query ? m_tgt_embed->weight.unsqueeze(0).repeat({ bs, 1, 1 }) : top_k_features;

    if (is_training())
    {
        refer_bbox = refer_bbox.detach();
        if (m_learnt_init_query)
            embeddings = embeddings.detach();
    }

    if (dn_embed.has_value())
        embeddings = torch::cat({ dn_embed.value(), embeddings}, 1);

    return { embeddings, refer_bbox, enc_bboxes, enc_scores };
}

std::tuple<torch::Tensor, torch::Tensor> RTDETRDecoderImpl::_generate_anchors(std::vector<std::vector<int>> shapes,
    torch::Tensor like_feats, float grid_size, float eps)
{
    std::vector<torch::Tensor> vc_anchors;
    auto t_ops = torch::TensorOptions(like_feats.dtype()).device(like_feats.device());
    for (int i = 0; i < shapes.size(); i++)
    {
        int h = shapes[i][0];
        int w = shapes[i][1];

        auto sy = torch::arange(h, t_ops);
        auto sx = torch::arange(w, t_ops);

        auto grid_yx = torch::meshgrid({ sy, sx }, "ij");
        auto grid_y = grid_yx[0];
        auto grid_x = grid_yx[1];

        auto grid_xy = torch::stack({ grid_x, grid_y }, -1);
        auto valid_WH = torch::tensor({ w, h }, t_ops);
        grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH;
       
        auto wh = torch::ones_like(grid_xy, t_ops) * grid_size * (std::pow(2, i));
        vc_anchors.push_back(torch::cat({ grid_xy, wh }, -1).view({ -1, h * w, 4 }));
    }

    torch::Tensor anchors = torch::cat(vc_anchors, 1);  // (1, h*w*nl, 4)
    torch::Tensor valid_mask = ((anchors > eps) & (anchors < (1 - eps))).all(-1, true);

    anchors = torch::log(anchors / (1 - anchors));
    float inf_value = std::numeric_limits<float>::infinity();
    anchors = anchors.masked_fill(~valid_mask, inf_value);
    return { anchors, valid_mask };

}
