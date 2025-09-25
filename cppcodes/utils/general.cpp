#include <filesystem>
#include <regex>
#include "general.h"

std::string get_root_path_string()
{
    std::filesystem::path exe_path = std::filesystem::absolute(std::filesystem::current_path());
#ifdef WIN32
    std::string root_path = exe_path.parent_path().parent_path().string(); //windows: ../cppcodes/build/Release, 如果设置Working dirctory = ${projectDir}, 就与Linux下一样
#else
    std::string root_path = exe_path.parent_path().parent_path().string();              //linux:  ../cppcodes/build
#endif
    std::cout << "The root dir: " << root_path << std::endl;
    return root_path;
}

void load_default_environment(const std::string& root_path, VariantConfigs& opts)
{
    std::string cfgs_opt_path = std::filesystem::path(root_path).append("cfgs").append("opt.yaml").string();
    if (!std::filesystem::exists(std::filesystem::path(cfgs_opt_path))) {
        LOG(WARNING) << "Can't found " << cfgs_opt_path << " , will use default value";
        opts = set_cfg_opt_default();
    }
    else {
        opts = load_cfg_yaml(cfgs_opt_path);
    }
}

#include "utils.h"

std::tuple<std::string, std::filesystem::file_time_type> found_newest_ptfile(const std::filesystem::path& path)
{
    int found = 0;
    std::string f_path = "";
    std::filesystem::file_time_type f_time;
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if (entry.is_directory())
        {
            auto [f, t] = found_newest_ptfile(entry.path());
            if (f != "")
            {
                if (t > f_time || f_path == "")
                {
                    f_path = f;
                    f_time = t;
                }
            }
        }
        else
        {
            auto fileext = entry.path().extension().string();
            if (fileext == ".pt")
            {
                if (found == 0)
                {
                    f_time = entry.last_write_time();
                    f_path = entry.path().string();
                }
                else
                {
                    if (entry.last_write_time() > f_time)
                    {
                        f_time = entry.last_write_time();
                        f_path = entry.path().string();
                    }
                }
                found += 1;
            }
        }
    }

    return std::make_tuple(f_path, f_time);
}

std::string get_last_run(std::string search_dir)
{
    auto root_ = std::filesystem::path(search_dir);
    auto [f, t] = found_newest_ptfile(root_);
    return f;
}


std::string increment_path(const std::string prj_and_name, bool exist_ok/* = true*/, std::string sep/* = ""*/)
{
    auto pth = std::filesystem::path(prj_and_name);
    if (!std::filesystem::exists(pth))   return prj_and_name;
    if (std::filesystem::exists(pth) && exist_ok) return prj_and_name;

    std::vector<int> indices;
    std::regex pattern(pth.stem().string() + sep + "(\\d+)");

    for (const auto& entry : std::filesystem::directory_iterator(std::filesystem::path(prj_and_name).parent_path())) 
    {
        if (entry.is_directory())
        {
            std::smatch match;
            std::string filename = entry.path().filename().string();
            if (std::regex_search(filename, match, pattern)) {
                indices.push_back(std::stoi(match[1].str()));
            }
        }
    }


    int n = 0;
    if (indices.empty())
        n = 1;
    else 
        n = (*std::max_element(indices.begin(), indices.end()) + 1);
    
    return (prj_and_name + sep + std::to_string(n));
}

torch::Tensor bbox_iou(torch::Tensor box1_, torch::Tensor box2_, bool is_xywh /*= false*/,
    bool GIoU /*= false*/, bool DIoU/*= false*/, bool CIoU /*= false*/, float eps /*= 1e-7*/) 
{
    //std::cout << "bbox_iou() box1_: " << box1_.sizes() << " box2_: " << box2_.sizes() << std::endl;

    torch::Tensor b1_x1, b1_y1, b1_x2, b1_y2;
    torch::Tensor b2_x1, b2_y1, b2_x2, b2_y2;
    torch::Tensor w1, h1, w2, h2;

    // 提取坐标    
    auto box1_chunks = box1_.chunk(4, -1);
    auto box2_chunks = box2_.chunk(4, -1);
    // 处理xywh格式的边界框     
    if (is_xywh)
    {   // 将box1和box2需要重新转换为xyxy坐标
        auto w1_half = box1_chunks[2] / 2;
        auto h1_half = box1_chunks[3] / 2;
        b1_x1 = box1_chunks[0] - w1_half;
        b1_y1 = box1_chunks[1] - h1_half;
        b1_x2 = box1_chunks[0] + w1_half;
        b1_y2 = box1_chunks[1] + h1_half;

        auto w2_half = box2_chunks[2] / 2;
        auto h2_half = box2_chunks[3] / 2;
        b2_x1 = box2_chunks[0] - w2_half;
        b2_y1 = box2_chunks[1] - h2_half;
        b2_x2 = box2_chunks[0] + w2_half;
        b2_y2 = box2_chunks[1] + h2_half;
    }
    else
    {
        b1_x1 = box1_chunks[0];
        b1_y1 = box1_chunks[1];
        b1_x2 = box1_chunks[2];
        b1_y2 = box1_chunks[3];

        b2_x1 = box2_chunks[0];
        b2_y1 = box2_chunks[1];
        b2_x2 = box2_chunks[2];
        b2_y2 = box2_chunks[3];
    }
    w1 = b1_x2 - b1_x1;
    h1 = (b1_y2 - b1_y1).clamp(eps);

    w2 = b2_x2 - b2_x1;
    h2 = (b2_y2 - b2_y1).clamp(eps);

    //std::cout << "b2_x1: " << b2_x1.sizes() << std::endl;
    //  交集区域 Intersection area
    auto inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) *
        (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0);
    //std::cout << "inter " << inter.sizes() << std::endl;
    auto union_ = ((w1 * h1 + w2 * h2) - inter) + eps;
    //std::cout << "union " << union_.sizes() << std::endl;
    // 计算IoU
    auto iou = inter / union_;

    if (CIoU || DIoU || GIoU) {
        auto cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1);
        auto ch = b1_y1.maximum(b2_y2) - b1_y1.minimum(b2_y1);

        if (CIoU || DIoU) {
            auto c2 = torch::pow(cw, 2) + torch::pow(ch, 2) + eps;  // 凸包对角线平方
            auto rho2 = (torch::pow((b2_x1 + b2_x2 - b1_x1 - b1_x2), 2) +
                torch::pow((b2_y1 + b2_y2 - b1_y1 - b1_y2), 2)) / 4;  // 中心点距离平方

            if (CIoU) 
            {
                torch::Tensor alpha;
                auto v = (4.f / std::pow(M_PI, 2)) * torch::pow((torch::atan(w2 / h2) - torch::atan(w1 / h1)), 2);
                {
                    torch::NoGradGuard no_grad;
                    alpha = v / (v - iou + (1 + eps));
                }
                return iou - (rho2 / c2 + v * alpha);  // CIoU
            }
            return iou - (rho2 / c2);  // DIoU
        }

        auto c_area = cw * ch + eps;  // 凸包区域
        return iou - (c_area - union_) / c_area;  // GIoU
    }
    return iou;  // 普通IoU
}


/*
    // python 原代码注释，坐标为xyxy格式
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
*/
torch::Tensor box_iou(const torch::Tensor & boxes1, const torch::Tensor & boxes2)
{
    auto box1 = boxes1.t();
    auto box2 = boxes2.t();

    auto box_area = [](torch::Tensor box){
        return (box[2] - box[0]) *(box[3] - box[1]);
    };

    auto area1 = box_area(box1);
    auto area2 = box_area(box2);

    /*
    // python 原代码
    // inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    */
    auto bbox_intersect = [](torch::Tensor box1, torch::Tensor box2) {
    // box1: [N, 4], box2: [N, 4]
        auto box1_expand_min = box1.index({torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(2)});
        auto box2_select_min = box2.index({torch::indexing::Slice(), torch::indexing::Slice(2)});
        auto box1_expand_max = box1.index({torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(0, 2)});
        auto box2_select_max = box2.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)});

        auto min_xy = torch::min(box1_expand_min, box2_select_min); //min(x2y2)
        auto max_xy = torch::max(box1_expand_max, box2_select_max); //max(x1y1)
        return  (min_xy - max_xy).clamp(0).prod(2); // [N,M]
        };
    
    auto inter = bbox_intersect(boxes1, boxes2);

    return inter/(area1.unsqueeze(-1) + area2 - inter);
}

torch::Tensor segment2box(torch::Tensor& segment, int width, int height)
{
    auto x_coords = segment.select(1, 0);
    auto y_coords = segment.select(1, 1);

    auto mask_x = (x_coords >= 0) * (x_coords <= width);
    auto mask_y = (y_coords >= 0) * (y_coords <= height); 
    auto mask = mask_x * mask_y;
    auto valid_x = torch::masked_select(x_coords, mask);
    auto valid_y = torch::masked_select(y_coords, mask);
    // 都超采样了，如果还没有足够多的点，说明大多数在有效区外，先进行面积筛选掉
    if(valid_x.size(0) > 10 && valid_y.size(0) > 10)
    {
        float x_min = torch::min(valid_x).item<float>();
        float x_max = torch::max(valid_x).item<float>();
        float y_min = torch::min(valid_y).item<float>();
        float y_max = torch::max(valid_y).item<float>();
        return torch::tensor({ x_min, y_min, x_max, y_max });
    }
    // 压入空矩形，后面通过与原始输入targets一对一比较筛选掉这些超区域的标签
    return torch::tensor({0,0,0,0});    
}


// 输入为[N][M,2] ==> [N][n=1000, 2]
std::vector<torch::Tensor> resample_segments(std::vector<torch::Tensor>& segments, int n /*= 1000*/) 
{
	std::vector<torch::Tensor> resampled_segments;

	for (int i = 0; i < segments.size(); ++i) 
	{
		auto s = segments[i];
		int n_pt = s.size(0);
		// 闭合多边形
		auto closed = torch::cat({ s, s.index({0}).unsqueeze(0) });	// n_pt+1
		int remainder = n % n_pt;
		int nstep = (n - remainder) / n_pt;
		std::vector<float> new_s;
		auto x_coords = closed.index({ torch::indexing::Slice(), 0 });
		auto y_coords = closed.index({ torch::indexing::Slice(), 1 });
		for (int j = 0; j < n_pt; j++)
		{
			float x1 = x_coords[j].item<float>();
			float x2 = x_coords[j + 1].item<float>();
			float y1 = y_coords[j].item<float>();
			float y2 = y_coords[j + 1].item<float>();

			if (j == (n_pt - 1))	// 最后一次，补齐余数
				nstep += remainder;

			float d_x = (x2 - x1) / float(nstep);
			float d_y = (y2 - y1) / float(nstep);

			for (int k = 0; k < nstep; k++)
			{
				new_s.push_back(x1 + d_x * k);
				new_s.push_back(y1 + d_y * k);
			}
		}

		auto xy_interp = torch::tensor(new_s).view({ -1, 2 });
		resampled_segments.push_back(xy_interp);
	}
	return resampled_segments;
}

ModelEMA::ModelEMA(std::shared_ptr<Model> ptr_model, float decay, int updates)
{
    decay_ = decay;
    updates_ = updates;
    auto model = ptr_model->get();
    // const std::string& yaml_file, int classes, int imagewidth, int imageheight, int channels, bool showdebuginfo
    ema_model_ = std::make_shared<Model>(model->cfgfile, model->n_classes, 
        model->image_width, model->image_height, model->n_channels, model->b_showdebug);
    (*ema_model_)->eval();

    for (auto& param : (*ema_model_)->parameters()) {
        param.set_requires_grad(false);
    }
}

void ModelEMA::update(torch::nn::Module& model)
{
    (*ema_model_)->to(model.parameters().begin()->device());

    {
        torch::NoGradGuard nograd;
        updates_ += 1;
        float d = decay_function(updates_);

        std::unordered_map<std::string, torch::Tensor> model_params;
        for (const auto& param : model.named_parameters())
        {
            model_params[param.key()] = param.value();
            // std::cout << "model_params key: " << param.key() << std::endl;
        }

        for (auto& param : (*ema_model_)->named_parameters())
        {
            std::string str_name = param.key();
            // std::cout << "ema_model_ key: " << str_name << std::endl;

            if (param.value().is_floating_point())
            {
                auto v = param.value().clone();
                v = v * d;
                v = v + (1.f - d) * model_params[str_name].detach();
                param.value().data().copy_(v.data());
            }
        }
    }
}
#include <fstream>
std::string getLastLine(const std::string& filename) 
{
    std::ifstream file(filename);
    if (!file.is_open()) 
        return ""; 
    file.seekg(-1, std::ios_base::end); 
    long pos = file.tellg(); 
    while (pos > 0 && file.get() != '\n') 
    { // 向上回溯找到第一个回车符
        --pos;
        file.seekg(pos, std::ios_base::beg); 
    }
    std::string lastLine;
    std::getline(file, lastLine); // 再读取下一行，就是最后一行了
    return lastLine;
}