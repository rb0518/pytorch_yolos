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
// 将Detect和Segment合并为一个类，统一batch数据处理函数，后续将is_segment和is_overlap传递给
// CustomExample::apply_batch()
// data = [1, 3, 640, 640]
// target = [N, 6]
// mask = [1 or N, 160 or 168, 160 or 168] ? ? ? cat后，要不与target相同，要不与 batch相同
// 两种方式对应的是overlap True or False
//
struct CustomExample{
	torch::Tensor data;
	torch::Tensor target;
	std::vector<std::string> path;
	std::vector<std::vector<float>> shape;
	torch::Tensor mask;
};
/*
	注意std::vector<T> ==> T
*/
struct CustomCollate : torch::data::transforms::Collation<CustomExample>{
    CustomExample apply_batch(std::vector<CustomExample> batch) override 
	{
		std::vector<torch::Tensor> datas, targets, masks;
		std::vector<std::string> paths;
		std::vector<std::vector<float>> shapes;
		bool is_segment = true;

		// 通过mask的sizes()大小判定是哪类数据，在非Segment数据中，mask插入的是torch::zeros({1});
		if(batch[0].mask.sizes().size() == 1)
			is_segment = false;
		
		if(is_segment)
		{
			bool allmask_dim0_is_one = true;
			for (auto& item : batch)
			{
				if (item.mask.size(0) != 1)
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
					if (allmask_dim0_is_one)		// 暂时这么处理，后续修改将overlap标志压入到 CustomExampleSeg 中
						masks.push_back(item.mask);
				}

				paths.push_back(item.path[0]);
				shapes.push_back(item.shape[0]);
				i += 1;
			}
		}
		else
		{
			int i = 0;
			for (auto& item : batch) {
				datas.push_back(item.data);
				masks.push_back(item.mask);
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
		}
		CustomExample ret = {torch::stack(datas), torch::cat(targets),paths, shapes, torch::cat(masks)};
		return ret;
    }
};

// ============================== class LoadImagesAndLabels ==========================
class LoadImagesAndLabels : public torch::data::Dataset<LoadImagesAndLabels, CustomExample>
{
public:
	LoadImagesAndLabels(std::string _path, VariantConfigs _hyp,
		int _img_size, bool _augment, bool _rect, bool _single_cls, int _stride, float _pad, 
		bool _is_segment = false, bool _is_overlap = false, int _downsample_ratio = 4);

	CustomExample get(size_t index);
	torch::optional<size_t> size() const override {
		return img_files.size();
	}

public:
	std::string		path;
	VariantConfigs	hyp;
	int				img_size = 640;
	bool			augment = false;
	bool			rect = false; 
	bool			single_cls = false;
	int				stride = 32; 
	float			pad = 0.0f; 

	bool			mosaic;
	std::vector<int> mosaic_border;

	std::vector<std::string> img_files;
	std::vector<std::string> label_files;

	std::string		images_dir;
	std::string		labels_dir;
	std::vector<int> indices;	// �洢���������

	bool is_segment = false;
	bool overlap = false;
	int downsample_ratio = 4;

	CustomExample get_detect(size_t index);
	std::tuple<cv::Mat, torch::Tensor> load_mosaic_detect(int index);

	CustomExample get_segment(size_t index);
	std::tuple<cv::Mat, torch::Tensor, std::vector<torch::Tensor>> load_mosaic_segment(int index);
};

// 更换为shared_ptr的原因是因为std::unique_ptr不利于每次epoch调用val时指针的传递
#if 1
using Dataloader_Custom = std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<LoadImagesAndLabels, CustomCollate>, torch::data::samplers::SequentialSampler>>;
#else
typedef torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<LoadImagesAndLabels, CustomCollate>, torch::data::samplers::SequentialSampler>  Dataloader_CutomeType;
//typedef torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<LoadImagesAndLabels, CustomCollate>, torch::data::samplers::RandomSampler>  Dataloader_CutomeType;
typedef std::shared_ptr<Dataloader_CutomeType> Dataloader_Custom;
#endif
std::tuple<Dataloader_Custom, int> create_dataloader(
						const std::string& path,
                        int imgsz,
						int nc,
                        int batch_size,
                        int stride,
                        VariantConfigs& opt,
                        VariantConfigs& hyp,
                        bool augment = false,
                        float pad = 0.0f,
						bool is_val = false,
						bool is_segment = false,
						bool is_overlap = false,
						int downsample_ratio = 4);
