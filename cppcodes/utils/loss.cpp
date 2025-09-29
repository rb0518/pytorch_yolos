#include "loss.h"
#include "general.h"
#include "utils.h"

//  c10::Dict<std::string, torch::IValue> hyp;
#include <torch/torch.h>
#include <cmath>

torch::Tensor crop_mask(torch::Tensor masks, torch::Tensor boxes)
{
    int n = masks.size(0);
    int h = masks.size(1);
    int w = masks.size(2);

    // 将boxes分割为四个坐标分量
    auto x1 = boxes.select(1, 0).unsqueeze(1).unsqueeze(2);  // x1 shape(1,1,n)
    auto y1 = boxes.select(1, 1).unsqueeze(1).unsqueeze(2);  // y1 shape(1,1,n)
    auto x2 = boxes.select(1, 2).unsqueeze(1).unsqueeze(2);  // x2 shape(1,1,n)
    auto y2 = boxes.select(1, 3).unsqueeze(1).unsqueeze(2);  // y2 shape(1,1,n)

    // 创建行和列的索引张量
    auto r = torch::arange(0, w).to(masks.device()).to(masks.dtype())
        .unsqueeze(0).unsqueeze(0);  // rows shape(1,1,w)
    auto c = torch::arange(0, h).to(masks.device()).to(masks.dtype())
        .unsqueeze(0).unsqueeze(2);  // cols shape(1,h,1)

    // 创建裁剪掩码
    auto mask = (r >= x1) * (r < x2) * (c >= y1) * (c < y2);

    // 应用掩码到输入mask
    return masks * mask;
}    

torch::Tensor single_mask_loss(torch::Tensor gt_mask, 
                              torch::Tensor pred, 
                              torch::Tensor proto, 
                              torch::Tensor xyxy, 
                              torch::Tensor area,
                              int nm) 
{
    // std::cout << "gt_mask " << gt_mask.sizes() << " pred " << pred.sizes() <<
    //     " proto: " << proto.sizes() << " xyxy " << xyxy.sizes() <<
    //     " area " << area.sizes() << std::endl;
    // 计算预测mask: (n,32) @ (32,80*80) -> (n,80*80) -> (n,80,80)
    auto pred_mask = torch::matmul(pred, proto.view({nm, -1})).view({-1, proto.size(1), proto.size(2)});
    //std::cout << "pred_mask" << pred_mask.sizes() << std::endl;
    // 计算二元交叉熵损失
    auto loss = torch::binary_cross_entropy_with_logits(pred_mask, gt_mask, {}, {}, at::Reduction::None);
    //std::cout << "loss " << loss.sizes() << std::endl;
    // 裁剪mask并计算归一化损失
    auto cropped_loss = crop_mask(loss, xyxy);  // 假设已实现crop_mask函数
    //std::cout << "cropped_loss" << cropped_loss.sizes() << std::endl;
    return (cropped_loss.mean({1, 2}) / area).mean();
}

ComputeLoss::ComputeLoss(std::shared_ptr<DetectImpl> m, VariantConfigs& _hyp, 
        bool _autobalance/* = false*/, bool _overlap /*= false*/)
{
    this->_device = m->parameters()[0].device();
    this->is_segment = m->is_segment;
    this->overlap = _overlap;
    std::cout << "is_segment : " << is_segment << std::endl;
    if (this->_device.type() == torch::kCPU)
        std::cout << "init compute loss device is cpu" << std::endl;

	if(_hyp.size())
		for (auto& [k, v] : _hyp)	
			hyp[k] = v;

    // Define criteria
    //std::cout<< "cls_pw " << std::get<float>(hyp["cls_pw"]) << " obj_pw " << std::get<float>(hyp["obj_pw"]) << std::endl;
    BCEcls = torch::nn::BCEWithLogitsLoss(
        torch::nn::BCEWithLogitsLossOptions().pos_weight(
            torch::tensor({ std::get<float>(hyp["cls_pw"]) }, _device)));

    BCEobj = torch::nn::BCEWithLogitsLoss(
        torch::nn::BCEWithLogitsLossOptions().pos_weight(
            torch::tensor({ std::get<float>(hyp["obj_pw"]) }, _device)));
    // 类未继承自torch::nn::Module, 暂时不用修改

    auto smooth_BCE = [&](float eps = 0.1){
        return std::make_tuple(1.0 - 0.5 * eps, 0.5*eps);
    };
    // Class label smoothing
    auto eps = 0.0f;
    if(hyp.count("label_smoothing") > 0) 
    {
        eps = std::get<float>(hyp["label_smoothing"]);
    }
    std::tie(cp, cn) = smooth_BCE(eps);

    // Focal loss 暂时不用   
    float g = std::get<float>(hyp["fl_gamma"]);
    //std::cout << "fl_gamma: " << g << std::endl;

    if (g > 0) 
    {
        //FocalLoss_cls = FocalLoss(BCEcls, g, 0.25, _device);
        //FocalLoss_obj = FocalLoss(BCEobj, g, 0.25, _device);
    }

    this->nl = m->nl;
    balance = (this->nl == 3) ? std::vector<double>{4.0, 1.0, 0.4}  // P3-P5
            : std::vector<double>{ 4.0, 1.0, 0.25, 0.06, 0.02 };    // P3-P7

    strides = m->stride;    // 初始化时，Detect中还未根据输入生成，所以不要用clone
  
    ssi = autobalance ? 1 : 0;

    this->gr = 1.0;
    this->autobalance = _autobalance;
    this->na = m->na;
    this->nc = m->nc;
    //anchors = m->named_buffers()["anchors"];
    if(m->is_segment)
    {
        std::shared_ptr<SegmentImpl> last_s = std::dynamic_pointer_cast<SegmentImpl>(m);
        this->nm = last_s->nm;
        std::cout << "last module return nm: " << nm << std::endl;
    }
    m_ptr = m;    
}

/*
//  %[parameters]:
//  p [][bs, anchor_num, grid_h, grid_w, nl]] = {[bs, 3, 80, 80, nl],
//                                              [bs, 3, 40, 40, nl],
//                                              [bs, 3, 20, 20, nl]};
//  targets [num_object, batch_index+class+xywh] = [nt, 6]
//  %[returns]:
//   (lcss + lbox + lobj) * bs 整个batch的总损失, 进行反射传播
//   cat(lbox, lobj, lcls, loss).detach() 回归损失、置信度损失，分类损失和总损失
*/
std::tuple<torch::Tensor, torch::Tensor> ComputeLoss::operator()(const std::vector<torch::Tensor>& p, const torch::Tensor& proto,
        torch::Tensor& targets, torch::Tensor& masks) 
{
    //std::cout << "compute_loss: " << proto.sizes() << " targets: " << targets.sizes() << " masks: " << masks.sizes() << std::endl;
    auto device = targets.device();
    int bs, nm, mask_h, mask_w;
    if(is_segment)
    {
        bs = proto.size(0);
        nm = proto.size(1);
        mask_h = proto.size(2);
        mask_w = proto.size(3);
        //std::cout << "bs: " << bs << " nm " << nm << " mask_h " << mask_h << " " << mask_w << std::endl;
    }
    overlap = false;
    // 
    auto lcls = torch::zeros({ 1 }, _device);   // class loss
    auto lbox = torch::zeros({ 1 }, _device);   // box loss
    auto lobj = torch::zeros({ 1 }, _device);   // object loss
    auto lseg = torch::zeros({ 1 }, _device);

    auto [tcls, tbox, indices, anchors_indices, tidxs, xywhn] = build_targets(p, targets);
    //std::cout << "build_targets over..." << std::endl;
    torch::Tensor tobj;
    for (int i = 0; i < p.size(); ++i) {
        auto pi = p[i];
        //std::cout << i << " pi: " << pi.sizes() << std::endl;
        auto [b, a, gj, gi] = indices[i];

        tobj = torch::zeros(pi.sizes().slice(0, 4),
            pi.options().device(_device));

        int n = b.size(0);
        if (n > 0) {        
            auto ps = pi.index({ b, a, gj, gi });
            std::vector<torch::Tensor> ps_split;
            if(is_segment)
                ps_split = ps.split({2, 2, 1, nc, nm}, 1);
            else
                ps_split = ps.split({2, 2, 1, nc}, 1);
            auto pxy = ps_split[0];
            auto pwh = ps_split[1];
            auto pcls = ps_split[3];
            pxy = pxy.sigmoid() * 2 - 0.5;
            pwh = (pwh.sigmoid() * 2).pow(2) * anchors_indices[i];
            auto pbox = torch::cat({pxy, pwh}, 1);  // predicted box

            auto iou = bbox_iou(pbox, tbox[i], true, false, false,true).squeeze();
            lbox += (1.0 - iou).mean();

            tobj.index_put_({ b, a, gj, gi },  (1.0 - gr) + gr * iou.detach().clamp(0).to(tobj.dtype()));

            // Classification
            if (nc > 1) {
                auto t = torch::full_like(pcls, 
                                cn).to(_device);
                t.index_put_({ torch::arange(n), tcls[i] }, cp);
                lcls += BCEcls->forward(pcls, t);
            }

            // Mask regression
            if(is_segment)
            {   
                auto pmask = ps_split[4];
                //std::cout << "pmask " << pmask.sizes() << std::endl;

                // 后续修改函数定义，不用const，目前不让改动是为了后续能够画图显示调试信息
                //auto mask_inter = masks.clone();
                auto mask_shape = masks.sizes();
                if(mask_shape[mask_shape.size()-2] != mask_h ||
                    mask_shape[mask_shape.size()-1] != mask_w)
                {
                    masks = torch::nn::functional::interpolate(
                        masks.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{mask_h, mask_w}).mode(torch::kNearest)).squeeze(0);                   
                }
                //std::cout << "masks: mask_h" << mask_h << " new masks: " << masks.sizes() << std::endl;
                auto marea = xywhn[i].index({torch::indexing::Slice(), 
                                            torch::indexing::Slice(2, torch::indexing::None)}).prod(1);
                auto scale = torch::tensor({mask_w, mask_h, mask_w, mask_h}, 
                                            masks.options().device());
                auto mxyxy = xywh2xyxy(xywhn[i] * scale);
                auto unique_b = std::get<0>(torch::_unique(b));
                for(int b_idx = 0; b_idx < unique_b.size(0); b_idx++)
                {
                    auto bi = unique_b[b_idx].item().toInt();
                    auto j = (b == bi);
                    torch::Tensor mask_gti;
                    if (overlap) {
                        mask_gti = torch::where(
                            masks[bi].unsqueeze(0) == tidxs[i].index({j.to(torch::kLong)}).view({-1, 1, 1}),
                            1.0, 0.0);
                        //std::cout << "overlap over..." << mask_gti.sizes() << std::endl;
                    } else 
                    {
                        auto temp = masks.index({tidxs[i]});
                        mask_gti = temp.index({j.to(torch::kLong)});
                        //std::cout << "no overlap over..." << mask_gti.sizes() << std::endl;   
                    }

                    lseg += single_mask_loss(mask_gti, 
                                            pmask.index({j.to(torch::kLong)}),
                                            proto[bi],
                                            mxyxy.index({j.to(torch::kLong)}),
                                            marea.index({j.to(torch::kLong)}),
                                            nm);     
                    //std::cout << "lseg: " << lseg << "  " << lseg.dtype() << std::endl;
                }
            }   // end for segment
        }
        torch::Tensor obji = BCEobj->forward(pi.index({ "...", 4 }), tobj);
        lobj += obji * balance[i];

        if (autobalance) {
            balance[i] = balance[i] * 0.9999 + 0.0001 / obji.detach().item<double>();
        }
    }

    if (autobalance) {
        auto base = balance[ssi];
        for (auto& x : balance) x /= base;
    }

    lbox *= std::get<float>(hyp["box"]);
    lobj *= std::get<float>(hyp["obj"]);
    lcls *= std::get<float>(hyp["cls"]);

    torch::Tensor loss;
    torch::Tensor ret_lossitems;

    if(is_segment)
    {
        lseg *= (std::get<float>(hyp["box"]) / bs);
        loss = (lbox + lobj + lcls + lseg);
        ret_lossitems = torch::cat({lbox, lobj, lcls, lseg}).detach();
    }
    else
    {
        bs = tobj.size(0);
        loss = (lbox + lobj + lcls);
        ret_lossitems = torch::cat({lbox, lobj, lcls, loss}).detach();  // 用loss占位，代替lseg，保持所有的操作是一致的
    }    
    auto ret_loss = loss * bs;
    return { ret_loss, ret_lossitems };
}
/*
// build_targets:
//    函数用于获得在训练时计算 loss 所需要的目标框，也即正样本。
// parameters:
// p: {[bs, 3, 80, 80, 85], [bs, 3, 40, 40, 85], [bs, 3, 20, 20, 85]}
// 正常的数据应该重载Dataset函数，targets将多张图合并[num_target(nt), image_idex, class+xywh] 
// xywh 归一[0, 1]
// return:
//      tcls: target所属class index
//      tbox xywh   xy为这个target对当前grid_cell左上角的偏移量
//      indices:    b target属于的image index
//                  a 表示target所使用的anchor index
//                  gj 经过筛选后确定某个target在网格中的 行偏移
//                  gi 网格左上角 列偏移
*/  
std::tuple<std::vector<torch::Tensor>,
    std::vector<torch::Tensor>,
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>,
    std::vector<torch::Tensor>,
    std::vector<torch::Tensor>, // tidxs
    std::vector<torch::Tensor>> // xywhn
    ComputeLoss::build_targets(const std::vector<torch::Tensor>& p, const torch::Tensor& _targets)  
{
    // na = 3, nt = 标签数
    int nt = _targets.size(0);    

    std::vector<torch::Tensor> tcls, tbox, anch;
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> indices;
    
    std::vector<torch::Tensor> tidxs;
    std::vector<torch::Tensor> xywhn;
    // 初始化增益系数
    auto gain = torch::ones({8}).to(_targets.device());
    auto ai = torch::arange({na}).to(_targets.device()).to(torch::kFloat)
        .view({ na, 1 }).repeat({ 1, nt });
    auto targets = _targets.clone();

    // 暂时未考虑overlap模式

    auto ti = torch::arange({nt}).to(_targets.device()).to(torch::kFloat)
        .view({1, nt}).repeat({na, 1});

    targets = torch::cat({ _targets.repeat({na, 1, 1}), 
                ai.index({"...", torch::indexing::None}),
                ti.index({"...", torch::indexing::None})}, 
                2);
    // [3, 242, 8]  ti [3, 252]
    // 定义网格偏移量，应该是多个网格等效进行插值偏移 jk, jm, lk, lm 中心+上下左右
    float g = 0.5;
    auto off = torch::tensor({  {0,0}, 
                                {1,0}, 
                                {0,1}, 
                                {-1,0}, 
                                {0,-1} }).to(_targets.device()).to(torch::kFloat) * g;

    for (int i = 0; i < this->nl; ++i) 
    {
        auto anchors = m_ptr->anchors_[i];    // 当前feature 对应的 anchor尺寸
        //std::cout << "test anchors: " << anchors.sizes() << " anchor: " << anchors << std::endl;
        // gain[2:6] = flow.tensor(p[i].shape, device=self.device)[[3, 2, 3, 2]].float()
        auto shape = p[i].sizes();
        gain[2] = static_cast<float>(shape[3]);
        gain[3] = static_cast<float>(shape[2]);
        gain[4] = static_cast<float>(shape[3]);
        gain[5] = static_cast<float>(shape[2]);

        auto t = targets * gain;   // [3, nt, 7] == new [3, 242, 8]
        torch::Tensor offsets = torch::zeros({ 1 });
        if (nt > 0) {
            // r [3, 242,2]
            auto r = t.index({"...", torch::indexing::Slice(4, 6)}) 
                            / anchors.index({torch::indexing::Slice(), torch::indexing::None});
            // [3, nt, 2] wh ratio

            auto [max_values, max_indices] = torch::max(r, 1.0 / r).max(2);
            auto j = max_values < std::get<float>(hyp["anchor_t"]);
            // 过滤掉负样本，t [3, nt, 7] ==> [j, 7], masked_select后只有一维了，所以要重新转换为[n, 7]的格式
            auto full_mask = j.unsqueeze(-1); //.expand(t.sizes());
            // fullmask [3, 242]
            t = torch::masked_select(t, full_mask);
            t = t.view({-1, 8});        // [370, 8]

            // 2. 网格偏移补偿
            auto gxy = t.index({torch::indexing::Slice(), torch::indexing::Slice(2, 4)});    //  gxy = t[:, 2:4]  # grid xy [n, 2]   n == t.shape[0]
            auto gxi = gain.index({torch::tensor({2, 3})}) - gxy;       // gxi --> 相对于右下角的坐标， gain[2, 3] wh

            // j, k = ((gxy % 1 < g) & (gxy > 1)).T
            // l, m = ((gxi % 1 < g) & (gxi > 1)).T
            // j, k, l, m 量[n]的张量
            // 假设gxy是二维浮点张量，g是标量阈值
            auto gxy_condition = (gxy.fmod(1) < g) & (gxy > 1);
            auto gxy_transposed = gxy_condition.t();  // 执行转置
            // 显式拆分转置后的张量
            //std::cout << "T转换 " << transposed.sizes() << std::endl;
            j = gxy_transposed[0];  // 获取第一行               
            auto k = gxy_transposed[1];  // 获取第二行
            //std::cout << "T j " << j.sizes() << "  k " << k.sizes() << std::endl;
            auto gxi_condition = (gxi.fmod(1) < g) & (gxi > 1);
            auto gxi_transposed = gxy_condition.t();
            auto l = gxi_transposed[0];
            auto m = gxi_transposed[1];
            
            // j = flow.stack((flow.ones_like(j), j, k, l, m))  [5, n]
            j = torch::stack({ torch::ones_like(j), j, k, l, m});//.unsqueeze(-1);

            // t = t.repeat((5, 1, 1))[j]   t [n, 7] == > [5, n, 7]
            //t = t.repeat({5, 1, 1}).masked_select(j).view({-1, 7});
            t = t.repeat({5, 1, 1}).index({j});
            {
                // 创建与gxy同形状的零张量并增加批次维度
                auto zeros = torch::zeros_like(gxy).unsqueeze(0);  // [None]操作
                // 为off添加中间维度实现广播
                auto expanded_off = off.unsqueeze(1);  // [:,None]操作
                // 执行广播加法
                auto offset_matrix = zeros + expanded_off;
                offsets = offset_matrix.index({j});
            }
        }
        else
        {   
            t = targets[0];
            offsets = torch::zeros({ 1 });  //通过[n,2]和[1]维大小来判定是否有值
        }

        // 3. 构建输出目标
        // b, c = t[:, :2].long().T
        auto t_chunks = t.chunk(4, 1);
        auto bc = t_chunks[0];
        auto gxy = t_chunks[1];
        auto gwh = t_chunks[2];
        auto at  = t_chunks[3];

        auto bc_transpose = bc.to(torch::kLong).t();
        auto b = bc_transpose[0];
        auto c = bc_transpose[1];
        auto at_transpose = at.to(torch::kLong).t();
        auto a = at_transpose[0];
        auto tidx = at_transpose[1];

        if (offsets.sizes().size() == 1)
        {
            offsets = torch::zeros_like(gxy);
            //std::cout << "no label , so offsets size [1] new: " << offsets.sizes() << std::endl;
        }

        auto gij = (gxy - offsets).to(torch::kLong);
        auto gij_transpose = gij.t();    //grid xy indices
        auto gi = gij_transpose[0]; 
        auto gj = gij_transpose[1]; 
       
        indices.push_back({b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1) });
        tbox.push_back(torch::cat({gxy-gij, gwh}, 1));
        anch.push_back(anchors.index({ a }));
        tcls.push_back(c);
        tidxs.push_back(tidx);
        xywhn.push_back(torch::cat({gxy, gwh}, 1) / gain.index({torch::tensor({2, 3, 4, 5})}));
    }

    return { tcls, tbox, indices, anch, tidxs, xywhn};   // C++11后支持列表初始化，可不用std::make_tuple
}
