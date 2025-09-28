#include "loss.h"
#include "general.h"

//  c10::Dict<std::string, torch::IValue> hyp;
#include <torch/torch.h>
#include <cmath>

ComputeLoss::ComputeLoss(std::shared_ptr<DetectImpl> m, VariantConfigs& _hyp, bool autobalance/* = false*/)
{
    this->_device = m->parameters()[0].device();
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
    this->autobalance = autobalance;
    this->na = m->na;
    this->nc = m->nc;
    //anchors = m->named_buffers()["anchors"];

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
std::tuple<torch::Tensor, torch::Tensor> ComputeLoss::operator()(const std::vector<torch::Tensor>& p,
    const torch::Tensor& targets) 
{
    auto device = targets.device();
    // 
    auto lcls = torch::zeros({ 1 }, _device);   // class loss
    auto lbox = torch::zeros({ 1 }, _device);   // box loss
    auto lobj = torch::zeros({ 1 }, _device);   // object loss

    /*
    *   对应着每个feature层，list_size=features_size，每层的n相对应
        tcls = {[n1], [n2], [n3]}
        tbox = {[n1, 4], [n2, 4], [n3, 4]}
        indices = {{[n1], [n1], [n1], [n1]}, {[n2], [n2], [n2], [n2], ...}
        anchors = {[n1, 2], [n2, 2], [n3, 2]}

     */
    //this->anchors = m_ptr->nam;
    //this->strides = m_ptr->stride;
    //std::cout << "p " << p[0].device().type() << " anchors: " << anchors.device().type() << " strides: " << strides.device().type() << std::endl;
    //std::cout << "targets " << targets.device().type() << std::endl;
    //std::cout << "m_ptr: " << this->m_ptr->named_buffers()["anchors"].device().type() << " " << this->m_ptr->stride.device().type() << std::endl;
    auto [tcls, tbox, indices, anchors_indices] = build_targets(p, targets);
    //std::cout << "build targets over..." << std::endl;
    // Losses
    torch::Tensor tobj;
    for (int i = 0; i < p.size(); ++i) {
        auto pi = p[i];
        auto [b, a, gj, gi] = indices[i];

        tobj = torch::zeros(pi.sizes().slice(0, 4),
            pi.options().dtype(pi.dtype()));

        int n = b.size(0);
        if (n > 0) {        
            auto ps = pi.index({ b, a, gj, gi });
            auto ps_split = ps.tensor_split({2, 4, 5}, 1);
            auto pxy = ps_split[0];
            auto pwh = ps_split[1];
            auto pcls = ps_split[3];
            pxy = pxy.sigmoid() * 2 - 0.5;
            pwh = (pwh.sigmoid() * 2).pow(2) * anchors_indices[i];
            auto pbox = torch::cat({pxy, pwh}, 1);  // predicted box

            auto iou = bbox_iou(pbox, tbox[i], true, false, false,true).squeeze();
            //std::cout << "bbox pbox out iou: " << iou.sizes() << std::endl;
            lbox += (1.0 - iou).mean();

            tobj.index_put_({ b, a, gj, gi },  (1.0 - gr) + gr * iou.detach().clamp(0).to(tobj.dtype()));

            // Classification
            if (nc > 1) {
                auto t = torch::full_like(ps.index({torch::indexing::Slice(), torch::indexing::Slice(5)}), 
                                cn).to(_device);
                t.index_put_({ torch::arange(n), tcls[i] }, cp);
                lcls += BCEcls->forward(ps.index({torch::indexing::Slice(), torch::indexing::Slice(5)}), t);
            }
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
    auto bs = tobj.size(0);

    auto loss = (lbox + lobj + lcls);
    auto ret_loss = loss * bs;

    auto ret_lossitems = torch::cat({lbox, lobj, lcls, loss}).detach();
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
    std::vector<torch::Tensor>>
    ComputeLoss::build_targets(const std::vector<torch::Tensor>& p, const torch::Tensor& _targets)  
{
    // na = 3, nt = 标签数
    int nt = _targets.size(0);    

    std::vector<torch::Tensor> tcls, tbox, anch;
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> indices;

    // 初始化增益系数
    auto gain = torch::ones({7}).to(_targets.device());
    auto ai = torch::arange({na}).to(_targets.device()).to(torch::kFloat)
        .view({ na, 1 }).repeat({ 1, nt });

    //std::cout << "ai: " << ai.sizes() << std::endl;    
    auto targets = _targets.clone();

    targets = torch::cat({ _targets.repeat({na, 1, 1}), 
                ai.index({"...", torch::indexing::None})}, 
                2);
    //std::cout << "targets clone: " << targets.sizes() << std::endl;
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

        auto t = targets * gain;   // [3, nt, 7]
        torch::Tensor offsets = torch::zeros({ 1 });
        if (nt > 0) {
            //
            auto r = t.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(4, 6)}) / anchors.index({torch::indexing::Slice(), torch::indexing::None});
            //std::cout << "r size: " << r.sizes() << std::endl;  // [3, nt, 2] wh ratio

            auto [max_values, max_indices] = torch::max(r, 1.0 / r).max(2);
            auto j = max_values < std::get<float>(hyp["anchor_t"]);
            
            // 过滤掉负样本，t [3, nt, 7] ==> [j, 7], masked_select后只有一维了，所以要重新转换为[n, 7]的格式
            auto full_mask = j.unsqueeze(-1); //.expand(t.sizes());
            if( (full_mask == true).sum().item().toInt() == 0)
            {
                //std::cout << "anchor_t: " << std::get<float>(hyp["anchor_t"]) << std::endl;    
                //std::cout << "===================>  full_mask == true " << (full_mask == true).sum().item().toInt() << std::endl;
            }
            t = torch::masked_select(t, full_mask);
            t = t.view({-1, 7});
            //std::cout << "filter t:  " << t.sizes() << " j " << j.sizes() << std::endl;

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
            //std::cout << "t : " << t.sizes() << " j " << j.sizes() << std::endl;

            {
                // 创建与gxy同形状的零张量并增加批次维度
                auto zeros = torch::zeros_like(gxy).unsqueeze(0);  // [None]操作
                // 为off添加中间维度实现广播
                auto expanded_off = off.unsqueeze(1);  // [:,None]操作
                // 执行广播加法
                auto offset_matrix = zeros + expanded_off;

                //std::cout << "offset_matrix " << offset_matrix.sizes() << std::endl;
                //offsets = offset_matrix.masked_select(j).view({-1, 2});
                offsets = offset_matrix.index({j});
                //std::cout << "offsets " << offsets.sizes() << std::endl;
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
        auto a  = t_chunks[3];

        auto bc_transpose = bc.to(torch::kLong).t();
        auto b = bc_transpose[0];
        auto c = bc_transpose[1];
        a = a.to(torch::kLong).view(-1);

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
    }

    return { tcls, tbox, indices, anch };   // C++11后支持列表初始化，可不用std::make_tuple
}
