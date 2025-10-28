#pragma once
#include <tuple>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <filesystem>

#include "yaml_load.h"
#include "utils.h"
#include <random>

#define USE_YOLOCustom 1

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
	//std::vector<std::vector<float>> shape;
	torch::Tensor ori_shape;
	torch::Tensor resized_shape;
	torch::Tensor mask;
};
/*
	注意std::vector<T> ==> T
*/
// 2025-10-18 根据Ultralytics YOLO最新代码中数据格式要求，先修改数据组织格式，后续重新编写Dataset代码
// IValue作为Dict的Key, Value在C++中有限定，可以通过c10::impl::GenericDict来构建，但在参数传递的时候
// 还有指针交换时，编译会报错，用std::map代替是为了保留代码，并让编译器通过
// using YoloCustomExample = c10::Dict<std::string, c10::IValue>
using YoloCustomExample = c10::Dict<std::string, torch::Tensor>;
struct YoloCustomCollate : torch::data::transforms::Collation<YoloCustomExample> 
{
public:	
	YoloCustomExample apply_batch(std::vector<YoloCustomExample> batch) override
	{
		std::vector<torch::Tensor> v_imgs;
		std::vector<torch::Tensor> v_cls;
		std::vector<torch::Tensor> v_bboxes;
		std::vector<torch::Tensor> v_batch_idx;
//		torch::List<std::string> v_im_files;	//IValue不能用，std::string不太好转换torch::Tensor，暂时不用
		/*
			后续可以用struct YoloCustomExample{
				vector<string> path,
				torch::Dict<string, Tensor> labels
			}; 来传递path
		*/
		std::vector<torch::Tensor> v_ori_shapes;
		std::vector<torch::Tensor> v_resized_shapes;
		std::vector<torch::Tensor> v_masks;
		for(int i_idx = 0; i_idx < batch.size(); i_idx++)
		{
			auto item = batch[i_idx];
			v_imgs.push_back(item.at("img"));
			v_cls.push_back(item.at("cls"));
			auto bboxes = item.at("bboxes");
			v_bboxes.push_back(bboxes);
			
			auto t_idx = item.at("batch_idx");
			t_idx.fill_(i_idx);
			v_batch_idx.push_back(t_idx);
			// 对其它如segment, pose数据的支持后续完成

			// auto im_file = item.at("im_file").toList();
			// v_im_files.push_back(im_file.get(0).toStringRef());
			v_ori_shapes.push_back(item.at("ori_shape"));
			v_resized_shapes.push_back(item.at("resized_shape"));
			if (item.contains("mask"))
			{
				v_masks.push_back(item.at("mask"));
			}
		}
		YoloCustomExample ret_example;
		//ret_example.insert({ "im_file", torch::IValue(v_im_files) });
		ret_example.insert( "ori_shape", torch::stack(v_ori_shapes));
		ret_example.insert( "resized_shape", torch::stack(v_resized_shapes));

		ret_example.insert( "img", torch::stack(v_imgs, 0));		// n:[3, 640, 640] ==> [n, 3, 640, 640];
		ret_example.insert( "cls", torch::cat(v_cls, 0));		// [[n1], [n2]...[nx] => [n1+n2+...+nx];
		ret_example.insert( "bboxes", torch::cat(v_bboxes, 0));
		ret_example.insert( "batch_idx", torch::cat(v_batch_idx, 0));

		if(v_masks.size())
			ret_example.insert( "target", torch::stack(v_masks));
		// 显式构造目标字典类型
		return ret_example;
	}
};

// ============================== class LoadImagesAndLabels ==========================
class LoadImagesAndLabels : public torch::data::Dataset<LoadImagesAndLabels, YoloCustomExample>
{
public:
	LoadImagesAndLabels(std::string _path, VariantConfigs* _args, int _stride = 32, bool _is_segment = false);			
	YoloCustomExample get(size_t index);

	torch::optional<size_t> size() const override {
		return img_files.size();
	}
public:
	std::string		path;
	VariantConfigs*	args;
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
	std::vector<int> indices;	// 生成乱序队列

	bool is_segment = false;
	bool overlap = false;
	int downsample_ratio = 4;

	CustomExample get_detect(size_t index);
	std::tuple<cv::Mat, torch::Tensor> load_mosaic_detect(int index);

	CustomExample get_segment(size_t index);
	std::tuple<cv::Mat, torch::Tensor, std::vector<torch::Tensor>> load_mosaic_segment(int index);
};

using Dataloader_Custom = std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<LoadImagesAndLabels, YoloCustomCollate>, torch::data::samplers::SequentialSampler>>;

std::tuple<Dataloader_Custom, int> create_dataloader(
						const std::string& path,
						VariantConfigs args,
                        int stride = 32,
						bool is_val = false,
						bool is_segment = false);						

inline void check_batchsize(VariantConfigs& args)
{
	int batchsize = std::get<int>(args["batch"]);
	if (batchsize <= 0 || batchsize > 32)
		args["batch"] = 16;
}

class DataloaderBase
{
public:
	explicit DataloaderBase(const std::string& root_path, VariantConfigs& _args_input, int stride = 32, 
		bool _is_val = false, bool is_segment = false);

	int get_total_samples(bool batch = false) {
		if (!batch)
			return total_numbers;
		int add_1 = total_numbers % batch_size;
		int num_batch = (total_numbers - add_1) / batch_size;
		if(add_1) num_batch += 1;
		std::cout << "total_numbers: " << total_numbers << " batch size: " << batch_size << " number of batch: " << num_batch << std::endl;
		return num_batch;
	}

	void cloase_mosaic(bool bclose);
	bool parse_data_yaml(const std::string& root);
	int get_batch_size() { return batch_size; }
public:
	Dataloader_Custom dataloader;
private:
	int total_numbers = 0;
	int batch_size = 16;
	bool is_val = false;
	std::string 	data_yaml_path;
	VariantConfigs  args;
	std::string		datasets_path;
	std::vector<std::string> names;
};

