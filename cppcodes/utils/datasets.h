#pragma once
#include <tuple>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <filesystem>

#include "yaml_load.h"
#include "utils.h"
#include <random>

int listallfiles_withsuffixes(const std::filesystem::path& path, 
	std::vector<std::string>& file_lists, std::vector<std::string> suffixes, bool enable_subfolder = false);


class LoadImagesAndLabels : public torch::data::Dataset<LoadImagesAndLabels>
{
public:
	explicit LoadImagesAndLabels(std::string _path, VariantConfigs _hyp,
		int _img_size = 640, int _batch_size = 16, bool _augment = false,
		bool _rect = false, bool _image_weights = false, bool _cache_images = false,
		bool _single_cls = false, int _stride = 32, float _pad = 0.0f, std::string _prefix = "");

	torch::data::Example<> get(size_t index) override;
	torch::optional<size_t> size() const override {
		return img_files.size();
	}

	std::vector<ExampleType> get_batch(c10::ArrayRef<size_t> indices) override
	{
		std::vector<torch::Tensor> imgs;
		std::vector<torch::Tensor> labels;
		//        std::cout << "indices: " << indices << std::endl;
		for (int i = 0; i < indices.size(); i++)
		{
			auto idx = indices[i];
			//std::cout << "idx : " << idx << std::endl;
			torch::data::Example<> sample = get(idx);
			auto img = sample.data;
			img = img.squeeze(0);
			//            std::cout << "image " << img.sizes() << std::endl;
			imgs.push_back(img);

			auto label = sample.target;
			//            std::cout << "label 0: " << label.sizes() << std::endl;
			if (label.sizes().size() == 2)
			{
				label.index_put_({ "...", 0 }, i);
				//std::cout << "label " << label.sizes() << std::endl;
				labels.push_back(label);
			}
		}

		auto batch_imgs = torch::stack(imgs, 0);
		//std::cout << " batch imgs: " << batch_imgs.sizes() << std::endl;
		// Concatenate labels along existing dimension

		torch::Tensor batch_labels;
		if (labels.size() > 1)
			batch_labels = torch::cat(labels, 0);
		else if (labels.size() == 1)
			batch_labels = labels[0];
		else
			batch_labels = torch::zeros({ 1 });
		//std::cout << " batch labels: " << batch_labels.sizes() << std::endl;

		std::vector<ExampleType> batch_ret;
		batch_ret.emplace_back(batch_imgs, batch_labels);
		return batch_ret;
	}

	// ???这个函数C++没法使用，估计得自己写Dataloader类才能实现
	//static torch::data::Example<> collate_fn(const std::vector<torch::data::Example<>>& batch){return batch;};
	std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::vector<int>>> 
	collate_fn(const std::vector<std::tuple<torch::Tensor, torch::Tensor, std::string, std::vector<int>>>& batch) 
	{
		std::cout << "collate_fn start..." << std::endl;

		std::vector<torch::Tensor> imgs, labels;
		std::vector<std::string> paths;
		std::vector<std::vector<int>> shapes_vec;
		
		for (const auto& item : batch) {
//			auto [img, label, path, shape] 
			imgs.push_back(std::get<0>(item));
			labels.push_back(std::get<1>(item));
			paths.push_back(std::get<2>(item));
			shapes_vec.push_back(std::get<3>(item));
		}

		// 添加label来自那张图像
		for (size_t i = 0; i < labels.size(); ++i) 
		{
			std::cout << i << " labels sizes: " << labels[i].sizes() << std::endl;
			if (labels[i].size(0) > 0) {
				labels[i].select(1, 0).fill_(i);
			}
		}

		auto img_stack = torch::stack(imgs, 0);
		auto label_cat = torch::cat(labels, 0);
		
		return std::make_tuple(img_stack, label_cat, paths, shapes_vec);
	}
     
    //static torch::data::Example<> collate_fn4(const std::vector<torch::data::Example<>>& batch){return batch;}; 
	static std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::vector<int>>> 
	collate_fn4(const std::vector<std::tuple<torch::Tensor, torch::Tensor, std::string, std::vector<int>>>& batch) 
	{
		std::vector<torch::Tensor> imgs, labels;
		std::vector<std::string> paths;
		std::vector<std::vector<int>> shapes;
		std::cout << "start collate_fn4: " << std::endl;
		for (const auto& item : batch) {
			imgs.push_back(std::get<0>(item));
			labels.push_back(std::get<1>(item));
			paths.push_back(std::get<2>(item));
			shapes.push_back(std::get<3>(item));
		}

		std::cout << shapes.size() << std::endl;
		const int n = shapes.size() / 4;
		std::vector<torch::Tensor> img4, label4;
		auto path4 = std::vector<std::string>(paths.begin(), paths.begin() + n);
		auto shapes4 = std::vector<std::vector<int>>(shapes.begin(), shapes.begin() + n);

		auto ho = torch::tensor({{0., 0, 0, 1, 0, 0}});
		auto wo = torch::tensor({{0., 0, 1, 0, 0, 0}});
		auto s = torch::tensor({{1, 1, 0.5, 0.5, 0.5, 0.5}});

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0.0, 1.0);

		// ??????????
		for (int i = 0; i < n; ++i) {
			int base_idx = i * 4;
			if (dis(gen) < 0.5) {
				// ???????
				auto img_interp = torch::nn::functional::interpolate(
					imgs[base_idx].unsqueeze(0).to(torch::kFloat),
					torch::nn::functional::InterpolateFuncOptions()
						.scale_factor(std::vector<double>{2.0, 2.0})
						.mode(torch::kBilinear)
						.align_corners(false)
				)[0].to(imgs[base_idx].dtype());
				img4.push_back(img_interp);
				label4.push_back(labels[base_idx]);
			} else {
				auto img_top = torch::cat({imgs[base_idx], imgs[base_idx + 1]}, 1);
				auto img_bottom = torch::cat({imgs[base_idx + 2], imgs[base_idx + 3]}, 1);
				auto img_combined = torch::cat({img_top, img_bottom}, 2);
				img4.push_back(img_combined);

				auto l1 = labels[base_idx];
				auto l2 = labels[base_idx + 1] + ho;
				auto l3 = labels[base_idx + 2] + wo;
				auto l4 = labels[base_idx + 3] + ho + wo;
				auto l_combined = torch::cat({l1, l2, l3, l4}, 0) * s;
				label4.push_back(l_combined);
			}
		}

		for (size_t i = 0; i < label4.size(); ++i) {
			if (label4[i].size(0) > 0) {
				label4[i].select(1, 0).fill_(i);
			}
		}

		return std::make_tuple(
			torch::stack(img4, 0),
			torch::cat(label4, 0),
			path4,
			shapes4
		);
	}
public:
	std::string		path;
	VariantConfigs	hyp;
	int				img_size = 640;
	int				batch_size = 16;
	bool			augment = false;
	bool			rect = false; 
	bool			image_weights = false;
	bool			catch_images = false;
	bool			single_cls = false;
	int				stride = 32; 
	float			pad = 0.0f; 
	std::string		prefix = "";

	bool			mosaic;
	std::vector<int> mosaic_border;

	std::vector<std::string> img_files;
	std::vector<std::string> label_files;

	std::string		images_dir;
	std::string		labels_dir;
	std::vector<int> indices;	// �洢���������

	std::tuple<cv::Mat, std::vector<int>, std::vector<int>> load_image(int index);
	std::tuple<cv::Mat, torch::Tensor> load_mosaic(int index);

	void load_xyhw_labels(const std::string& filename, std::vector<BBox_xyhw>& bboxs);

	std::tuple<cv::Mat, torch::Tensor> random_perspective(cv::Mat img, torch::Tensor targets,
		int degrees, float translate, float scale,
		int shear, float perspective, std::vector<int> border);
};

#if 0
std::pair<torch::data::DataLoaderBase<LoadImagesAndLabels>, LoadImagesAndLabels>  create_dataloader(const std::string& path,
                        int imgsz,
                        int batch_size,
                        int stride,
                        VariantConfigs& opt,
                        VariantConfigs& hyp,
                        bool augment = false,
                        bool cache = false,
                        float pad = 0.0f,
                        bool rect = false,
                        int rank = -1,
                        int world_size = 1,
                        int workers = 8,
                        bool image_weights = false,
                        bool quad = false,
                        const std::string& prefix = "") ;
#endif