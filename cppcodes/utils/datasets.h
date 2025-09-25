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

// ========================  Collect for Dectect =================================
// path和shape定义为vector是为了维持 vector<T> =>T，两个变量在mosaic模式下不起作用
struct CustomExample{
	torch::Tensor data;
	torch::Tensor target;
	std::vector<std::string> path;
	std::vector<std::vector<float>> shape;
};
/*
	注意std::vector<T> ==> T
*/
struct CustomCollate : torch::data::transforms::Collation<CustomExample>{
    CustomExample apply_batch(
        std::vector<CustomExample> batch) override {
        std::vector<torch::Tensor> datas, targets;
		std::vector<std::string> paths;
		std::vector<std::vector<float>> shapes;
		int i = 0;
        for (auto& item : batch) {
			datas.push_back(item.data);

			auto label = item.target;
			if (label.sizes().size() == 2 && label.size(0)!=0)
			{
				label.index_put_({ "...", 0 }, i);
	            targets.push_back(item.target);
			}

			paths.push_back(item.path[0]);
			shapes.push_back(item.shape[0]);
			i+=1;
        }
		CustomExample ret = {torch::stack(datas), torch::cat(targets),paths, shapes};
        return ret;
    }
};

// ========================  Collect for Segment =================================
/*
	data	= [1, 3, 640, 640]
	target	= [N, 6]
	mask	= [1 or N, 160 or 168, 160 or 168] ??? cat后，要不与target相同，要不与 batch相同
			两种方式对应的是overlap True or False
*/
struct CustomExampleSeg {
	torch::Tensor data;
	torch::Tensor target;
	std::vector<std::string> path;
	std::vector<std::vector<float>> shape;
	torch::Tensor mask;
};
/*
	注意std::vector<T> ==> T
*/
struct CustomCollateSeg : torch::data::transforms::Collation<CustomExampleSeg > {
	CustomExampleSeg apply_batch(std::vector<CustomExampleSeg> batch) {
		std::vector<torch::Tensor> datas, targets, masks;
		std::vector<std::string> paths;
		std::vector<std::vector<float>> shapes;
		bool allmask_dim0_is_one = true;
		for(auto& item : batch)
		{
			if(item.mask.size(0) != 1)
			{
				allmask_dim0_is_one = false;
				break;
			}
		}

		int i = 0;
		for (auto& item : batch) {
			datas.push_back(item.data);
			auto label = item.target;
			if (label.sizes().size() == 2 && label.size(0) != 0)
			{
				label.index_put_({ "...", 0 }, i);
				targets.push_back(item.target);
				masks.push_back(item.mask);			
			}
			else
			{// masks必须保持与targets同步压入数据，是否需要else里压入空图作为背景，后续根据loss里数据需求分析后添加
				// masks是根据overlap bool值变化，false 一种是跟masks保持dim(0)的一致, true 保持跟 datas一致
				if(allmask_dim0_is_one)		// 暂时这么处理，后续修改将overlap标志压入到 CustomExampleSeg 中
					masks.push_back(item.mask);
			}

			paths.push_back(item.path[0]);
			shapes.push_back(item.shape[0]);
			i += 1;
		}
		CustomExampleSeg ret = { torch::stack(datas, 0), torch::cat(targets, 0),paths, shapes, torch::cat(masks, 0)};
		return ret;
	}
};


// ============================== class LoadImagesAndLabels ==========================
class LoadImagesAndLabels : public torch::data::Dataset<LoadImagesAndLabels, CustomExample>
{
public:
	explicit LoadImagesAndLabels(std::string _path, VariantConfigs _hyp,
		int _img_size = 640, int _batch_size = 16, bool _augment = false,
		bool _rect = false, bool _image_weights = false, bool _cache_images = false,
		bool _single_cls = false, int _stride = 32, float _pad = 0.0f, std::string _prefix = "");

	CustomExample get(size_t index);
	torch::optional<size_t> size() const override {
		return img_files.size();
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

	std::tuple<cv::Mat, torch::Tensor> load_mosaic(int index);
};

// ============================== class LoadImagesAndLabelsAndMasks ==========================
class LoadImagesAndLabelsAndMasks : public torch::data::Dataset<LoadImagesAndLabelsAndMasks, CustomExampleSeg>
{
public:	
	explicit LoadImagesAndLabelsAndMasks(std::string _path, VariantConfigs _hyp,
		int _img_size = 640, int _batch_size = 16, bool _augment = false,
		bool _rect = false, bool _image_weights = false, bool _cache_images = false,
		bool _single_cls = false, int _stride = 32, float _pad = 0.0f, std::string _prefix = "");

	CustomExampleSeg get(size_t index);
	torch::optional<size_t> size() const override {
		return img_files.size();
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
	std::vector<int> indices;	

	bool overlap = false;
	int downsample_ratio = 4;
	std::tuple<cv::Mat, torch::Tensor, std::vector<torch::Tensor>> load_mosaic(int index);
};
