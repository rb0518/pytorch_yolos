#include <filesystem>
#include <random>
#include <algorithm> // std::shuffle
#include <fstream>
#include <thread>

#include "datasets.h"
#include "utils.h"
#include "general.h"
#include "augmentations.h"

#include "plots.h"

std::vector<std::string> img_format = { ".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".webp", ".mpo" }; //acceptable image suffixes
std::vector<std::string> vid_formats = { ".mov", ".avi", ".mp4", ".mpg", ".mpeg", ".m4v", ".wmv", ".mkv" };  // acceptable video suffixes

int listallfiles_withsuffixes(const std::filesystem::path& path, std::vector<std::string>& file_lists, 
	std::vector<std::string> suffixes, bool enable_subfolder /*= false*/)
{
    int oldfound = file_lists.size();
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if (entry.is_directory() && enable_subfolder)
        {
            listallfiles_withsuffixes(entry.path(), file_lists, suffixes, enable_subfolder);
        }
		else
		{
			auto fileext = entry.path().extension().string();
			if (std::count(suffixes.begin(), suffixes.end(), fileext) > 0)
			{
				file_lists.emplace_back(entry.path().string());
			}
		}
    }

    return file_lists.size() - oldfound;
}

// 读取图像，并返回原始尺寸和放缩后尺寸
std::tuple<cv::Mat, std::vector<int>, std::vector<int>> load_image(std::string filename, int img_size, bool augment)
{
	cv::Mat img = cv::imread(filename);
	if (img.empty())
		LOG(ERROR) << "load image: " << filename << " failed.";

	int h0 = img.rows;
	int w0 = img.cols;

	float r = float(img_size) / float(std::max(h0, w0));
	auto interpolation = r < 1.0 && augment ? cv::INTER_AREA : cv::INTER_LINEAR;
	cv::resize(img, img, cv::Size(int(w0 * r), int(h0 * r)), 0.0, 0.0, interpolation);
	std::vector<int> hw0 = { h0, w0 };				// image size数据格式传递统一为[height, width]
	std::vector<int> hw = { img.rows, img.cols };
	return std::make_tuple(img, hw0, hw);
}

std::vector<std::vector<float>> readlabels_fromfile(const std::string& filename)
{
	std::vector<std::vector<float>> result;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return result;
    }

    std::string line;
    while (std::getline(file, line)) 
	{
        std::vector<float> lineFloats;
        std::istringstream iss(line);
        float num;
        
        while (iss >> num) {
            lineFloats.push_back(num);
        }
        
        if (!lineFloats.empty()) {
            result.push_back(lineFloats);
        }
    }
    
    file.close();
    return result;
}

bool load_xyhw_labels(const std::string& filename, std::vector<std::vector<float>>& bboxs)
{
	bboxs = readlabels_fromfile(filename);

	if (bboxs.size() == 0 || bboxs[0].size() != 5)
	{
		std::cout << "read data error: data size != 5." << std::endl;
		return false;
	}

	return true;
}

bool read_segment_labels(const std::string& filename, std::vector<std::vector<float>>& bboxs, std::vector<std::vector<float>>& segments)
{
	std::vector<std::vector<float>> datas = readlabels_fromfile(filename);
	bboxs.clear();
	segments.clear();
	if (datas.size() == 0 || datas[0].size() <= 5 || (datas[0].size() % 2) != 1)
	{
		std::cout << "read segments label data type wrong..." << std::endl;
		return false;
	}

	for (int i = 0; i < datas.size(); i++)
	{
		int ncount = (datas[i].size() - 1) / 2;
		float minx, miny, maxx, maxy;
		for (int j = 0; j < ncount; j++)
		{
			float x = datas[i][1 + j * 2];
			float y = datas[i][1 + j * 2 + 1];
			if (j == 0)
			{
				minx = x;
				miny = y;
				maxx = minx;
				maxy = miny;
			}

			minx = std::min(minx, x);
			maxx = std::max(maxx, x);
			miny = std::min(miny, y);
			maxy = std::max(maxy, y);
		}
		float x_c = (minx + maxx) / 2;
		float y_c = (miny + maxy) / 2;
		float w = maxx - minx;
		float h = maxy - miny;
		float cls = datas[i][0];
		bboxs.emplace_back(std::vector<float>({ cls, x_c, y_c, w, h }));
		segments.emplace_back(datas[i].begin() + 1, datas[i].begin() + 1 + ncount * 2);
	}

	return true;
}

/*
* 扣图方法，实现图像增强：随机抽取满足条件的label，在指定的区域粘贴上副本，并将标签数据label和mask数据添加到当前标签组中
*/
void copy_past(cv::Mat& im, torch::Tensor& labels, std::vector<torch::Tensor>& segments, double p = 0.5)
{
	int h = im.rows;
	int w = im.cols;
	int n = labels.size(0);
	TORCH_CHECK(segments.size() == n, "Segments count must match labels count");

	// 计算两个框的
	auto fn_bbox_ioa = [](const cv::Rect& box1, const cv::Rect& box2) {     // 计算交集     
		int x1 = std::max(box1.x, box2.x);
		int y1 = std::max(box1.y, box2.y);
		int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
		int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
		if (x2 <= x1 || y2 <= y1) return 0.0;
		int intersection_area = (x2 - x1) * (y2 - y1);
		int union_area = box1.area() + box2.area() - intersection_area;
		return static_cast<double>(intersection_area) / union_area;
		};

	cv::Mat output = im.clone();
	if (p > 0.f && n > 0)
	{
		auto indices = random_queue(n);
		int select_size = int(float(n) * p);
		std::vector<std::vector<cv::Point>> contours;
		for (int i = 0; i < select_size; i++)
		{
			auto label = labels[indices[i]];
			auto segment = segments[indices[i]];

			float x1 = label[1].item<float>();
			float y1 = label[2].item<float>();
			float x2 = label[3].item<float>();
			float y2 = label[4].item<float>();
			// python             box = w - l[3], l[2], w - l[1], l[4]
			auto box1 = cv::Rect(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
			auto box2 = cv::Rect(cv::Point2f(w-x2, y1), cv::Point2f(w-x1, y2));

			auto ioa = fn_bbox_ioa(box1, box2);
			if (ioa < 0.30)
			{
				// 添加new label to labels;
				std::vector<float> new_lb = { label[0].item<float>(), w - x2, y1, w - x1, y2 };
				auto new_lb_tensor = torch::tensor(new_lb);
				torch::cat({ labels, new_lb_tensor }, 0);

				// 添加new segment to segments;
				// segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
				std::vector<float> new_seg;
				auto x_coords = segment.index({ torch::indexing::Slice(), 0 });
				auto y_coords = segment.index({ torch::indexing::Slice(), 1 });
				std::vector<cv::Point> contour;
				for (int j = 0; j < segment.size(0); j++)
				{
					new_seg.push_back(w - x_coords[j].item<float>());
					new_seg.push_back(y_coords[j].item<float>());
					contour.push_back(cv::Point(w - x_coords[j].item<float>(), y_coords[j].item<float>()));
				}
				auto new_seg_tensor = torch::tensor(new_seg).view({ -1, 2 });
				segments.push_back(new_seg_tensor);
				contours.push_back(contour);
			}
		}
		// 完成所有操作后，画出mask图像
		cv::Mat mask = cv::Mat::zeros(im.size(), CV_8UC1);
		cv::drawContours(mask, contours, -1, cv::Scalar(255), cv::FILLED);
		cv::flip(im, output, 1);
		cv::flip(mask, mask, 1);

		output.copyTo(im, mask);

		cv::imshow("output", output);
		cv::imshow("mask", mask);
		cv::imshow("im", im);

		cv::waitKey();
		cv::destroyAllWindows();
	}
}

cv::Mat polygon2mask(cv::Size img_size, torch::Tensor& segment, int color = 1, int downsample_ratio = 1)
{
	cv::Mat mask = cv::Mat::zeros(img_size, CV_8UC1);
	int ratio = std::max(1, downsample_ratio);
	std::vector<cv::Point> polygons;
	auto x = segment.select(1, 0);
	auto y = segment.select(1, 1);

	for (int i = 0; i < segment.size(0); i++)
	{
		polygons.push_back(cv::Point(x[i].item().toFloat(), y[i].item().toFloat()));
	}
	cv::fillPoly(mask, polygons, cv::Scalar(color));

	//cv::imshow("poly fill", mask);

	int nh = img_size.height / ratio;
	int nw = img_size.width / ratio;
	cv::resize(mask, mask, cv::Size(nw, nh));
	return mask;
}

/*
	2025-10-11 发现不论是tensor.fliplr()还是torch::flip(tensor, {2})，都未实现mask图像的翻转
	统一为返回cv::Mat数据，统一翻转后再进行图像统一处理
*/
#ifndef USE_TORCH_FLIP
std::vector<cv::Mat> polygons2masks(cv::Size img_size, std::vector<torch::Tensor> segments, int color, int downsample_ratio = 1)
{
	std::vector<cv::Mat> masks;
	for(int i = 0; i < segments.size(); i++)
	{
		auto s = segments[i];
		cv::Mat tmp_mat = polygon2mask(img_size, s, color, downsample_ratio);
		masks.push_back(tmp_mat);
	}
	return masks;
}
#else
torch::Tensor polygons2masks(cv::Size img_size, std::vector<torch::Tensor> segments, int color, int downsample_ratio = 1)
{
	std::vector<torch::Tensor> masks;
	for(int i = 0; i < segments.size(); i++)
	{
		auto s = segments[i];
		cv::Mat tmp_mat = polygon2mask(img_size, s, color, downsample_ratio);
		auto tmp_tensor = torch::from_blob(tmp_mat.data, {tmp_mat.rows, tmp_mat.cols }, torch::kByte).clone();
		masks.push_back(tmp_tensor);
	}
	torch::Tensor ret;
	if(masks.size() > 1)
		ret = torch::stack(masks, 0);
	else if(masks.size() == 1) 
		ret = masks[0].unsqueeze(0);	// [1, 640, 640]
	else
	{
		int nh = img_size.height / downsample_ratio;
		int nw = img_size.width / downsample_ratio;				
		ret = torch::zeros({1, nh, nw},  torch::TensorOptions().dtype(torch::kByte));
	}
	return ret;
}
#endif
/*
	overlap方式，将所有的标签画在一张图上，根据尺寸大小排序，并给出权值(从1开始)
	返回mask tensor, 为了与非overlap保持一致，返回[1, img_size, img_size]
	同时返回labels对应面积的排序
	2025-10-11测试发现torch::flip对mask图像不能进行翻转，所以修改为返回cv::Mat
*/
#ifndef USE_TORCH_FLIP
std::tuple<cv::Mat, torch::Tensor> polygons2masks_overlap(cv::Size img_size, std::vector<torch::Tensor> segments, int downsample_ratio = 1)
{
	// 发现按原代码转换，最大面积的object没有mask，暂时找不到原因
	std::vector<int64_t> areas;
	std::vector<std::vector<cv::Point>> polygons;
	for (int i = 0; i < segments.size(); i++)
	{
		auto s = segments[i];
		std::vector<cv::Point> polygon;
		auto x = s.select(1, 0);
		auto y = s.select(1, 1);

		for (int i = 0; i < s.size(0); i++)
		{
			polygon.push_back(cv::Point(int(x[i].item().toFloat()), int(y[i].item().toFloat())));
		}
		polygons.push_back(polygon);
		areas.push_back(cv::contourArea(polygon));
	}
	// 按面积大小排序
	auto sort_tensor_1d = [](const torch::Tensor& input, bool descend = false) {
		AT_ASSERT(input.dim() == 1, "input Tensor dim not equil 1");
		auto ascending_indices = torch::argsort(input);
		if (descend)
			return torch::flip(ascending_indices, 0);
		return ascending_indices;
		};

	auto descend_indices = sort_tensor_1d(torch::tensor(areas, torch::TensorOptions().dtype(torch::kLong)));
	
	int ratio = std::max(1, downsample_ratio);
	std::cout << segments.size() << " areas " << areas.size() << " segments: " << segments.size() << std::endl;
	// 根据面积排序将图像填充到masks中
	cv::Mat cv_mask = cv::Mat::zeros(img_size, CV_8UC1); 
	for (int i = 0; i < descend_indices.size(0); i++)
	{
		if(i > 254) break;	// 丢掉不能显示的数据

		int idx = descend_indices[i].item().toInt();
		std::cout << i << " descend: " << idx << std::endl;
		cv::fillPoly(cv_mask, polygons[idx], cv::Scalar(std::min(255, i+1)));
	}
	cv::resize(cv_mask, cv_mask, cv::Size(img_size.width / ratio, img_size.height / ratio));
	return std::make_tuple(cv_mask, descend_indices);
}
#else
std::tuple<torch::Tensor, torch::Tensor> polygons2masks_overlap(cv::Size img_size, std::vector<torch::Tensor> segments, int downsample_ratio = 1)
{
	// 发现按原代码转换，最大面积的object没有mask，暂时找不到原因
	std::vector<int64_t> areas;
	std::vector<std::vector<cv::Point>> polygons;
	for (int i = 0; i < segments.size(); i++)
	{
		auto s = segments[i];
		std::vector<cv::Point> polygon;
		auto x = s.select(1, 0);
		auto y = s.select(1, 1);

		for (int i = 0; i < s.size(0); i++)
		{
			polygon.push_back(cv::Point(int(x[i].item().toFloat()), int(y[i].item().toFloat())));
		}
		polygons.push_back(polygon);
		areas.push_back(cv::contourArea(polygon));
	}
	// 按面积大小排序
	auto sort_tensor_1d = [](const torch::Tensor& input, bool descend = false) {
		AT_ASSERT(input.dim() == 1, "input Tensor dim not equil 1");
		auto ascending_indices = torch::argsort(input);
		if (descend)
			return torch::flip(ascending_indices, 0);
		return ascending_indices;
		};

	auto descend_indices = sort_tensor_1d(torch::tensor(areas, torch::TensorOptions().dtype(torch::kLong)));
	
	int ratio = std::max(1, downsample_ratio);
	std::cout << segments.size() << " areas " << areas.size() << " segments: " << segments.size() << std::endl;
	// 根据面积排序将图像填充到masks中
	cv::Mat cv_mask = cv::Mat::zeros(img_size, CV_8UC1); 
	for (int i = 0; i < descend_indices.size(0); i++)
	{
		int idx = descend_indices[i].item().toInt();
		std::cout << i << " descend: " << idx << std::endl;
		cv::fillPoly(cv_mask, polygons[idx], cv::Scalar(std::min(255, i+1)));
	}
	cv::resize(cv_mask, cv_mask, cv::Size(img_size.width / ratio, img_size.height / ratio));
	torch::Tensor masks = torch::from_blob(cv_mask.data, {cv_mask.rows, cv_mask.cols }, torch::kByte).clone();
	return std::make_tuple(masks, descend_indices);
}
#endif
std::tuple<std::string, std::string> Init_ImageAndLabel_List(std::string& path, std::vector<std::string>& img_files, std::vector<std::string>& label_files)
{
	std::string images_dir;
	std::string labels_dir;
	auto get_labelpath = [&](std::string _path) {
			auto img_path = std::filesystem::path(_path);
			if(std::filesystem::is_directory(img_path) && std::filesystem::exists(img_path))
			{ 
				images_dir = path;
				std::string tmp = img_path.filename().string();		//such as: "train2017"
				std::cout << "path: " << path << std::endl;
				std::cout << "data name: " << tmp << std::endl;
				std::cout << "data path: " << img_path.parent_path().parent_path().string() << std::endl;

				labels_dir = img_path.parent_path().parent_path().append("labels").append(tmp).string();

				if (false == std::filesystem::exists(std::filesystem::path(labels_dir)))
					LOG(ERROR) << "Labels folder: " << labels_dir << " not exists.";

			}
		};

	std::vector<std::string> list_imgs;
	auto parent_path = std::filesystem::path(path).parent_path().string();
	bool b_save_filter = true;

	auto read_files_from_txt = [&](const std::string& filename, std::vector<std::string>& lists){
			std::ifstream file(filename);
			std::string line;
			if(file.is_open())
			{
				while(std::getline(file, line))
				{
					if(!line.empty() && line.back() == '\n')
						line.pop_back();
						
					if(!line.empty())
						lists.push_back(std::filesystem::path(parent_path).append(line).string());
				}
			}
			else{
				LOG(WARNING) << "open file: " << filename << " error!" << std::endl;
			}
		};

	if (std::filesystem::is_directory(std::filesystem::path(path)))
	{
		auto tmp_dirname = std::filesystem::path(path).stem().string();
		auto img_path_filter = std::filesystem::path(parent_path).append(tmp_dirname + ".filter");

		if(std::filesystem::exists(std::filesystem::path(img_path_filter)))
		{
			std::cout << "check filter file: " << img_path_filter.string() <<" Replace the original: " << path << std::endl;	
			read_files_from_txt(img_path_filter.string(), list_imgs);
			images_dir = parent_path;
			labels_dir = std::filesystem::path(parent_path).parent_path().append("labels/" + tmp_dirname).string();
			//std::cout << "convert labels_dir: " << labels_dir << std::endl;
			path = img_path_filter.string();
			//std::cout << "new path: " << path << " sub: " << path.substr(this->path.length() - 7) << std::endl;

			b_save_filter = false;
		}
		else
		{
			listallfiles_withsuffixes(std::filesystem::path(path), list_imgs, img_format);
			get_labelpath(path);
		}
	}
	else
	{	// 从txt文件读取，每行为图像:"./images/train2017/000000109622.jpg",根目录是coco
		auto img_path_filter = std::filesystem::path(path + ".filter");
		if(std::filesystem::exists(img_path_filter))
		{
			// 该文件是第一次数据集扫描生成的，已经整理过的数据，可以大大幅降低后续训练时载入时间
			std::cout << "check filter file: " << img_path_filter.string() <<" Replace the original: " << path << std::endl;	
			path = path + ".filter";
			b_save_filter = false; 
		}
		//std::cout << "datasets image root: " << parent_path << std::endl;
		read_files_from_txt(path, list_imgs);
		//std::cout << "total read images name count: " << list_imgs.size() << std::endl;
		if(list_imgs.size())
		{
			std::string first_name = list_imgs[0];
			std::string tmp = std::filesystem::path(first_name).parent_path().stem().string();
			labels_dir = std::filesystem::path(parent_path).append("labels/" + tmp).string();
		}
		else
			labels_dir = parent_path;
	}
	
	// 构造img_files和label_files队列，并随机打乱队列
	label_files.clear();
	img_files.clear();

	if(!std::filesystem::is_directory(std::filesystem::path(path))
		&& path.substr(path.length() - 7) == ".filter")
	{	// 读取以前的数据集，保存过的文件
		b_save_filter = false;

		for (int i = 0; i < list_imgs.size(); i++)
		{
			std::string item = list_imgs[i];
			std::string imgfile_stem = std::filesystem::path(item).stem().string();

			auto labelfile = std::filesystem::path(labels_dir).append(imgfile_stem + ".txt").string();
			if( i % 100 == 0){
				std::cout << "\x1b[2K\r" << "get label file lists: " << i << " | " << list_imgs.size() << " ";
				std::cout << std::flush;
			}	
			img_files.emplace_back(item);
			label_files.emplace_back(labelfile);
		}
		std::cout << "\x1b[2K\r" << "get label file lists: " << label_files.size() << " | " << list_imgs.size() << " " << std::endl;
	}
	else
	{
		// 对大数据集，这样操作太慢了，改为多线程处理，或者单独修改.txt文件
		auto check_fileexists = [&](int start_id, int end_id, const std::vector<std::string>& source_list, std::vector<std::string>& imgs, std::vector<std::string>& labels){
				for(int i = start_id; i < end_id; i++)
				{
					std::string item = source_list[i];
					std::string imgfile_stem = std::filesystem::path(item).stem().string();
					auto labelfile = std::filesystem::path(labels_dir).append(imgfile_stem + ".txt").string();
					if( start_id  == 0){
						std::cout << "\x1b[2K\r" << "scan files: " << end_id << " | " << i << " ";
						std::cout << std::flush;
					}
					if(std::filesystem::exists(std::filesystem::path(item)) && std::filesystem::exists(std::filesystem::path(labelfile)))
					{
						imgs.emplace_back(item);
						labels.emplace_back(labelfile);
					}
				}
			};  // end of lambda: check_fileexists
		const int thread_count = 8;
		std::vector<std::string> temp_imgs[thread_count];
		std::vector<std::string> temp_labels[thread_count];
		int temp_steps = list_imgs.size() / thread_count;
		std::thread thread_filter[thread_count];

		for(int threadid = 0; threadid < thread_count; threadid++)
		{
			int start_idx = threadid * temp_steps;
			int end_idx = std::min(int(list_imgs.size()), (start_idx + temp_steps));
			thread_filter[threadid] = std::thread(check_fileexists, start_idx, end_idx, std::ref(list_imgs), 
											std::ref(temp_imgs[threadid]), std::ref(temp_labels[threadid]));
		}
		for(int threadid = 0; threadid < thread_count; threadid++) 
			thread_filter[threadid].join();

		for(int v_id = 0; v_id < thread_count; v_id++)
		{
			for(int i = 0; i < temp_imgs[v_id].size(); i++)
			{
				img_files.emplace_back(temp_imgs[v_id][i]);
				label_files.emplace_back(temp_labels[v_id][i]);
			}
		}
		// std::cout << "total images: " << img_files.size() << std::endl;
	}
	if(b_save_filter)
	{	// 保存至原文件目录下
		auto save_path = path+".filter";
		std::ofstream ofs(save_path);
		if(ofs.is_open())
		{
			for(int i =0; i < img_files.size(); i++)
			{
				auto img_file_rel = std::filesystem::relative(
					std::filesystem::path(img_files[i]),
					std::filesystem::path(parent_path)
				).string();
				std::cout << "\x1b[2K\r" << "scan files: " << img_files.size() << " | " << i << " ";
				std::cout << std::flush;
				ofs << img_file_rel << "\n";
			}
			ofs.close();
		}
		std::cout << "save all exists files in file: " << save_path << std::endl;
	}

	return std::make_tuple(images_dir, labels_dir);
}
//========================   LoadImagesAndLabels ========================
LoadImagesAndLabels::LoadImagesAndLabels(std::string _path, VariantConfigs _hyp,
	int _img_size, bool _augment, bool _rect, bool _single_cls, int _stride, float _pad,
	bool _is_segment, bool _is_overlap, int _downsample_ratio)
	: img_size(_img_size), augment(_augment), single_cls(_single_cls), stride(_stride), pad(_pad), path(_path)
	, is_segment(_is_segment), overlap(_is_overlap), downsample_ratio(_downsample_ratio)
{
	std::cout << "LoadImagesAndLabels img_size " << img_size;
	rect = /*image_weights ? false :*/ _rect;
	mosaic = augment == true && rect == false;
	//mosaic = false;
	int half_size = static_cast<int>(std::trunc(float(img_size) / 2.0));
	mosaic_border = { -half_size, -half_size };

	if(_hyp.size())
		for (auto& [k, v] : _hyp)	
			hyp[k] = v;

	std::tie(images_dir, labels_dir) = Init_ImageAndLabel_List(_path, img_files, label_files);

	indices = random_queue(img_files.size());
}

CustomExample LoadImagesAndLabels::get(size_t index)
{
	if(0 == index)
	{
		indices = random_queue(img_files.size());
	}
	
	if(is_segment)
		return get_segment(index);
	return get_detect(index);
}

CustomExample LoadImagesAndLabels::get_detect(size_t index)
{
	auto indices_index = indices[index];
	auto need_mosaic = this->mosaic&& random_uniform() < std::get<float>(hyp["mosaic"]);

	cv::Mat img;
	torch::Tensor labels;
	torch::Tensor img_tensor;
	std::vector<std::string> paths;
	std::vector<std::vector<float>> shapes;
	if (need_mosaic)
	{
		paths.push_back("");
		//std::vector<float> tmp = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}  // 后三个分别对应 is_segment, overlap, downsample_ratio
		shapes.push_back(std::vector<float>(9, 0.f));
		std::tie(img, labels) = load_mosaic_detect(indices_index);
		if (random_uniform() < std::get<float>(hyp["mixup"]))
		{
			auto [img2, labels2] = load_mosaic_detect(int(random_uniform(0, indices.size() - 1)));

			auto r = random_beta(8.0f, 8.0f);
			cv::Mat img1_float, img2_float;
			
			img.convertTo(img1_float, CV_32F);
			img2.convertTo(img2_float, CV_32F);

			//两幅图像进融合
			cv::Mat mixed_img = img1_float * r + img2_float * (1 - r);

			// 融合图像返回到统一返回图像对象
			mixed_img.convertTo(img, img.type());
			// 两次label数据合并
			labels = torch::cat({ labels, labels2 }, 0);
		}
	}
	else
	{
		std::vector<int> hw0, hw;
		std::string filename = img_files[indices_index];
		paths.emplace_back(filename);

		std::tie(img, hw0, hw) = load_image(filename, img_size, augment);

		std::vector<float> ratio, pad;
		std::tie(img, ratio, pad)= letterbox(img, std::make_pair(img_size, img_size),
			cv::Scalar(114, 114, 114), false, false, augment, stride);

		// (w, h, w_r, h_r, w_p, h_p)
		shapes.emplace_back(std::vector<float>({ float(hw0[1]), float(hw0[0]), ratio[0], ratio[1], pad[0], pad[1], 0.f, 0.f, 0.f }));
	
		std::vector<std::vector<float>> bboxs;
		load_xyhw_labels(label_files[indices_index], bboxs);
		int labels_num = bboxs.size();
		labels = torch::zeros({ labels_num, 5 }).to(torch::kFloat32);
		if (labels.sizes().size() == 1 || labels_num == 0)
		{
			LOG(WARNING) << "read boxs number 0, file: " << label_files[indices_index];
		}

		// yolo标注格式 [image_index = (0,后期注入), cls_index, x, y, w, h ]
		for (int j = 0; j < labels_num; j++)
		{
			// 根据python原代码Yolov5-master中targets首位空为0，由后期注入labels数据对应batch
			labels[j].index_put_({ torch::indexing::Slice() }, torch::tensor(bboxs[j]));
		}
		auto w = ratio[0] * hw[1];
		auto h = ratio[1] * hw[0];

		auto select_xyhw = labels.index({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) });
		auto converted = xywhn2xyxy(select_xyhw, ratio[0] * hw[1], ratio[1] * hw[0], pad[0], pad[1]);
		labels.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) }, converted);
	}

	if (augment)
	{	
		/*
		if(false == need_mosaic)
			std::tie(img, labels) = random_perspective(img, labels,
				int(std::get<float>(hyp["degrees"])), std::get<float>(hyp["translate"]),
				std::get<float>(hyp["scale"]), int(std::get<float>(hyp["shear"])),
				std::get<float>(hyp["perspective"]), mosaic_border);
		*/
	
		augment_hsv(img, std::get<float>(hyp["hsv_h"]),
							std::get<float>(hyp["hsv_s"]),
							std::get<float>(hyp["hsv_v"]));
	}

	if(labels.size(0))
	{
		labels.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, 5)},
			xyxy2xywh(labels.index({torch::indexing::Slice(), torch::indexing::Slice(1, 5)}))
		);
		// labels[:,[2,4]] /= img.shape[0]
		labels.index_put_({ torch::indexing::Slice(), torch::tensor({2, 4})}, 
			labels.index({ torch::indexing::Slice(), torch::tensor({2, 4})})  / img.rows
			);
		// labels[:,[1,3]] /= img.shaoe[1]
		labels.index_put_({ torch::indexing::Slice(), torch::tensor({1, 3})}, 
			labels.index({ torch::indexing::Slice(), torch::tensor({1, 3})}) / img.cols
			);		
	}

	if (augment)
	{
		if(random_uniform() < (std::get<float>(hyp["flipud"])))
		{
			cv::flip(img, img, 0);
			labels.index_put_({torch::indexing::Slice(), 2},
				1 - labels.index({torch::indexing::Slice(), 2})
				);
		}
		if (random_uniform() < (std::get<float>(hyp["fliplr"])))
		{
			cv::flip(img, img, 1);
			labels.index_put_({torch::indexing::Slice(), 1},
				1 - labels.index({torch::indexing::Slice(), 1})
				);
		}
	}
	
	// label [nt,5] ==> [nt, 6]
	torch::Tensor labels_out = torch::zeros({labels.size(0), 6});
	if(labels.size(0))
	{
		labels_out.index_put_(
			{torch::indexing::Slice(), torch::indexing::Slice(1)},
			labels);
	}

	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);	// BGR==>RGB
	img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte)
                                   .permute({2, 0, 1});	// HWC ==> CHW
	// {tensor, tensor, vector<string>, vector<vector<float> 
	//		=> CollectBatch input: {vector<tensor>, vector<tensor>, vector<vector<string>>, vector<vector<vector<float>>>
	//		=> CollectBatch output: {tensor, tensor, vector<string>, vector<vector<float>
	CustomExample ret = {img_tensor.clone(), labels_out.clone(), paths, shapes, torch::zeros({1})};
	return ret;
}

std::tuple<cv::Mat, torch::Tensor> LoadImagesAndLabels::load_mosaic_detect(int index)
{
	int x = mosaic_border[0];
	auto yc = int(random_uniform(-x, 2 * img_size + x));
	auto xc = int(random_uniform(-x, 2 * img_size + x));

	cv::Mat img4 = cv::Mat(img_size*2, img_size*2, CV_8UC3, cv::Scalar(114,114,114));
	torch::Tensor labels4;

	std::vector<int> img_idx;
	img_idx.emplace_back(index);
	std::vector<torch::Tensor> labels;
	for (int i = 0; i < 3; i++)
	{
		int random_choice_idx = int(random_uniform(0, indices.size() - 1));
		img_idx.emplace_back(indices[random_choice_idx]);
	}

	for (int i = 0; i < 4; i++)
	{
		auto [img, hw0, hw] = load_image(img_files[img_idx[i]], img_size, augment);
		int x1a, y1a, x2a, y2a;
		int x1b, y1b, x2b, y2b;
		int h = hw[0];	// 统一为xyhw
		int w = hw[1];
		int s = img_size;

		if (i == 0) {  // top left
			x1a = std::max(xc - w, 0);
			y1a = std::max(yc - h, 0);
			x2a = xc;
			y2a = yc;
			x1b = w - (x2a - x1a);
			y1b = h - (y2a - y1a);
			x2b = w;
			y2b = h;
		}
		else if (i == 1) {  // top right
			x1a = xc;
			y1a = std::max(yc - h, 0);
			x2a = std::min(xc + w, s * 2);
			y2a = yc;
			x1b = 0;
			y1b = h - (y2a - y1a);
			x2b = std::min(w, x2a - x1a);
			y2b = h;
		}
		else if (i == 2) {  // bottom left
			x1a = std::max(xc - w, 0);
			y1a = yc;
			x2a = xc;
			y2a = std::min(s * 2, yc + h);
			x1b = w - (x2a - x1a);
			y1b = 0;
			x2b = w;
			y2b = std::min(y2a - y1a, h);
		}
		else if (i == 3) {  // bottom right
			x1a = xc;
			y1a = yc;
			x2a = std::min(xc + w, s * 2);
			y2a = std::min(s * 2, yc + h);
			x1b = 0;
			y1b = 0;
			x2b = std::min(w, x2a - x1a);
			y2b = std::min(y2a - y1a, h);
		}
		
		// 拷贝数据至目标roi区域
		cv::Mat roi = img4(cv::Rect(x1a, y1a, x2a - x1a, y2a - y1a));
		img(cv::Rect(x1b, y1b, x2b - x1b, y2b - y1b)).copyTo(roi);

		auto padw = x1a - x1b;
		auto padh = y1a - y1b;

		// ��ȡlabels, ???��ʱδ��segment
		std::vector<std::vector<float>> bboxs;
		load_xyhw_labels(label_files[img_idx[i]], bboxs);
		int labels_num = bboxs.size();
		torch::Tensor label_tensor = torch::zeros({ labels_num, 5 }).to(torch::kFloat32);
		if (label_tensor.sizes().size() == 1 || labels_num == 0)
		{
			LOG(WARNING) << "read boxs number 0, file: " << label_files[img_idx[i]];
		}

		// yolo��Ҫ�����ݸ�ʽ [image_index = (0,��ʱ), cls_index, x, y, w, h ]
		for (int j = 0; j < labels_num; j++)
		{
			// ����python������ԣ�Yolov5-master��targets���ݲ����˸������ݡ�����batch��ָ��������ͼ
			label_tensor[j].index_put_({ torch::indexing::Slice() }, torch::tensor(bboxs[j]));
		}
		auto select_xyhw = label_tensor.index({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) });
		auto converted = xywhn2xyxy(select_xyhw, w, h, padw, padh);
		label_tensor.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) }, converted);
		labels.emplace_back(label_tensor);
		//std::cout << "     read " << i << " labels: " << labels[i].sizes() << std::endl;
	}

	// Concat labels
	labels4 = torch::cat(labels, 0);
	//std::cout << "labels4: " << labels4.sizes() << std::endl;

	auto [aug_img4, aug_labels, _] = random_perspective(img4, labels4, 
			{},		// 保持segment版本一致，添加空队列
			int(std::get<float>(hyp["degrees"])), std::get<float>(hyp["translate"]),
			std::get<float>(hyp["scale"]), int(std::get<float>(hyp["shear"])),
			std::get<float>(hyp["perspective"]), mosaic_border);

	return std::make_tuple(aug_img4, aug_labels);
}

//---------------------  load for segment
CustomExample LoadImagesAndLabels::get_segment(size_t index)
{
	auto indices_index = indices[index];
	auto need_mosaic = this->mosaic && random_uniform() < std::get<float>(hyp["mosaic"]);
	cv::Mat img;
	torch::Tensor labels;
	torch::Tensor img_tensor;
	std::vector<std::string> paths;
	std::vector<std::vector<float>> shapes;
	std::vector<torch::Tensor> segments;

	torch::Tensor masks = torch::zeros({ 1 });	// 返回的masks[1, h, w] or [n, h, w]
	int padw, padh, w, h;
	if (need_mosaic)
	{
		paths.push_back("");
		std::vector<float> tmp = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f};
		if(this->overlap)	
			tmp.push_back(1.0f);
		else
			tmp.push_back(0.0f);
		tmp.push_back(float(downsample_ratio));
		shapes.push_back(tmp);
		std::tie(img, labels, segments) = load_mosaic_segment(indices_index);
		if (random_uniform() < std::get<float>(hyp["mixup"]))
		{
			auto [img2, labels2, segments2] = load_mosaic_segment(int(random_uniform(0, indices.size() - 1)));
			//两幅图像进融合 mixup
			auto r = random_beta(8.0f, 8.0f);
			cv::Mat img1_float, img2_float;
			img.convertTo(img1_float, CV_32F);
			img2.convertTo(img2_float, CV_32F);
			cv::Mat mixed_img = img1_float * r + img2_float * (1 - r);
			mixed_img.convertTo(img, img.type());
			// 两次label数据合并
			if(labels.size(0) == 0||labels2.size(0) == 0)
				std::cout << ColorString("labels is empty: ", "R") << " labels: " << labels.sizes() << " labels2: " << labels2.sizes() << std::endl;
			labels = torch::cat({ labels, labels2 }, 0);
			for (auto seg : segments2)
				segments.push_back(seg);
		}
	}
	else
	{
		std::vector<int> hw0, hw;
		std::string filename = img_files[indices_index];
		paths.emplace_back(filename);
		std::tie(img, hw0, hw) = load_image(filename, img_size, augment);

		std::vector<float> ratio, pad;
		std::tie(img, ratio, pad) = letterbox(img, std::make_pair(img_size, img_size),
			cv::Scalar(114, 114, 114), false, false, augment, stride);
		w = int(ratio[0] * hw[1]);
		h = int(ratio[1] * hw[0]);
		padw = int(pad[0]);
		padh = int(pad[1]);

		// (w, h, w_r, h_r, w_p, h_p)
		std::vector<float> tmp = {float(hw0[1]), float(hw0[0]), ratio[0], ratio[1], pad[0], pad[1]};
		if(this->overlap)	
			tmp.push_back(1.0f);
		else
			tmp.push_back(0.0f);
		shapes.emplace_back(tmp);
		std::vector<std::vector<float>> bboxs;
		std::vector<std::vector<float>> lbs_segments;
		read_segment_labels(label_files[indices_index], bboxs, lbs_segments);

		int labels_num = bboxs.size();
		labels = torch::zeros({ labels_num, 5 }).to(torch::kFloat32);
		if (labels.sizes().size() == 1 || labels_num == 0)
		{
			LOG(WARNING) << "read boxs number 0, file: " << label_files[indices_index];
		}

		for (int j = 0; j < labels_num; j++)
		{
			auto seg_tmp = torch::tensor(lbs_segments[j]).view({-1, 2});
			seg_tmp = xyn2xy(seg_tmp, w, h, padw, padh);
			segments.push_back(seg_tmp);
			labels[j].index_put_({ torch::indexing::Slice() }, torch::tensor(bboxs[j]));
		}
		auto select_xyhw = labels.index({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) });
		auto converted = xywhn2xyxy(select_xyhw, w, h, pad[0], pad[1]);
		labels.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) }, converted);
	}
	/*
	auto img_show = img.clone();
	cv::rectangle(img_show, cv::Point(padw, padh), cv::Point(padw + w, padh + h),
		cv::Scalar(220, 0, 0), 2);
	for (int i = 0; i < labels.size(0); i++)
	{
		float x1 = labels[i][1].item().toFloat();
		float x2 = labels[i][3].item().toFloat();
		float y1 = labels[i][2].item().toFloat();
		float y2 = labels[i][4].item().toFloat();
		cv::rectangle(img_show, cv::Point(x1, y1), cv::Point(x2, y2),
			cv::Scalar(220, 220, 0), 2);
	}
	cv::imshow("img", img_show);
	*/
	std::vector<cv::Mat> cv_masks;
	if(labels.size(0))
	{
		labels.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, 5)},
			xyxy2xywh(labels.index({torch::indexing::Slice(), torch::indexing::Slice(1, 5)}))
		);
		// labels[:,[2,4]] /= img.shape[0]
		labels.index_put_({ torch::indexing::Slice(), torch::tensor({2, 4})}, 
			labels.index({ torch::indexing::Slice(), torch::tensor({2, 4})})  / img.rows
			);
		// labels[:,[1,3]] /= img.shaoe[1]
		labels.index_put_({ torch::indexing::Slice(), torch::tensor({1, 3})}, 
			labels.index({ torch::indexing::Slice(), torch::tensor({1, 3})}) / img.cols
			);		

		if(overlap)
		{
			torch::Tensor tmp_indices;
			cv::Mat cv_mask;
			std::tie(cv_mask, tmp_indices) = polygons2masks_overlap(img.size(), segments, downsample_ratio);
			cv_masks.push_back(cv_mask);
			// masks = masks.unsqueeze(0);
			labels = labels.index_select(0, tmp_indices);
			if(labels.size(0) > 255)	// maske采用8UC1，只能提供255个数据，0为背景
				labels = labels.index({torch::indexing::Slice(0, 255)});
		}
		else{
			cv_masks = polygons2masks(img.size(), segments, 1, downsample_ratio);
		}
	}

	if (augment)
	{
		/*
		*	python 代码： 是外部库 
		*	img, labels = self.albumentations(img, labels)
		*		nl = len(labels)  # update after albumentations
		*/
		// HSV color-space
		augment_hsv(img, std::get<float>(hyp["hsv_h"]),
			std::get<float>(hyp["hsv_s"]),
			std::get<float>(hyp["hsv_v"]));

		if (random_uniform() < (std::get<float>(hyp["flipud"])))
		{
			cv::flip(img, img, 0);
			labels.index_put_({ torch::indexing::Slice(), 2 },
				1 - labels.index({ torch::indexing::Slice(), 2 })
			);
			for(int mat_idx = 0; mat_idx < cv_masks.size(); mat_idx++)
			{
				cv::flip(cv_masks[mat_idx], cv_masks[mat_idx], 0);
			}
		}
		if (random_uniform() < (std::get<float>(hyp["fliplr"])))
		{
			cv::flip(img, img, 1);
			labels.index_put_({ torch::indexing::Slice(), 1 },
				1 - labels.index({ torch::indexing::Slice(), 1 })
			);
			for(int mat_idx = 0; mat_idx < cv_masks.size(); mat_idx++)
			{
				cv::flip(cv_masks[mat_idx], cv_masks[mat_idx], 1);
			}
		}
	}

	// label [nt,5] ==> [nt, 6]
	torch::Tensor labels_out = torch::zeros({ labels.size(0), 6 });
	if (labels.size(0))
	{
		labels_out.index_put_(
			{ torch::indexing::Slice(), torch::indexing::Slice(1) },
			labels);
	}
	else{
		if(overlap)
		{	// 如果没有label数据，overlap方式下需要插入一张背景图
			int h = img.rows / downsample_ratio;
			int w = img.cols / downsample_ratio;
			cv::Mat cv_mask = cv::Mat::zeros(cv::Size(w, h), CV_8UC1); 
			cv_masks.push_back(cv_mask);
		}
	}

	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte).permute({2, 0, 1});

	// 构造masks
	if(overlap)
	{
		masks = torch::from_blob(cv_masks[0].data, {cv_masks[0].rows, cv_masks[0].cols}, torch::kByte).unsqueeze(0);
	}
	else
	{
		if(cv_masks.size() == 0)
		{	// 如果没有，压入一个[0, h, w]，占位用，确保后续跟labels一样，能够通过torch::cat
			masks = torch::zeros({0, img.rows/downsample_ratio, img.cols/downsample_ratio}, torch::kByte);
		}
		else
		{
			std::vector<torch::Tensor> vec_masks;
			for(int mat_idx = 0; mat_idx < cv_masks.size(); mat_idx++)
			{
				auto tmp_tensor = torch::from_blob(cv_masks[mat_idx].data, {cv_masks[mat_idx].rows, cv_masks[mat_idx].cols }, torch::kByte).clone();
				vec_masks.push_back(tmp_tensor);
			}
			if(vec_masks.size() > 1)
				masks = torch::stack(vec_masks, 0);
			else if(vec_masks.size() == 1) 
				masks = vec_masks[0].unsqueeze(0);	// [1, 640, 640]
			else
			{
				masks = torch::zeros({0, img.rows/downsample_ratio, img.cols/downsample_ratio}, torch::kByte);
			}
		}
	}

	CustomExample ret;	// test code
	ret.data = img_tensor.clone();	
	ret.target = labels_out.clone();
	ret.mask = masks.clone();
	ret.path = paths;
	ret.shape = shapes;
	return ret;
};

std::tuple<cv::Mat, torch::Tensor, std::vector<torch::Tensor>> 
	LoadImagesAndLabels::load_mosaic_segment(int index)
{
	int x = mosaic_border[0];
	auto yc = int(random_uniform(-x, 2 * img_size + x));
	auto xc = int(random_uniform(-x, 2 * img_size + x));

	cv::Mat img4 = cv::Mat(img_size*2, img_size*2, CV_8UC3, cv::Scalar(114,114,114));
	torch::Tensor labels4;
	std::vector<torch::Tensor> segments4;
	std::vector<int> img_idx;
	img_idx.emplace_back(index);
	std::vector<torch::Tensor> labels;
	std::vector<torch::Tensor> segments;	// 直接压入到segments4中???
	for (int i = 0; i < 3; i++)
	{
		int random_choice_idx = int(random_uniform(0, indices.size() - 1));
		img_idx.emplace_back(indices[random_choice_idx]);
	}

	for (int i = 0; i < 4; i++)
	{
		auto [img, hw0, hw] = load_image(img_files[img_idx[i]], img_size, augment);
		int x1a, y1a, x2a, y2a;
		int x1b, y1b, x2b, y2b;
		int h = hw[0];	// 统一为xyhw
		int w = hw[1];
		int s = img_size;

		if (i == 0) {  // top left
			x1a = std::max(xc - w, 0);
			y1a = std::max(yc - h, 0);
			x2a = xc;
			y2a = yc;
			x1b = w - (x2a - x1a);
			y1b = h - (y2a - y1a);
			x2b = w;
			y2b = h;
		}
		else if (i == 1) {  // top right
			x1a = xc;
			y1a = std::max(yc - h, 0);
			x2a = std::min(xc + w, s * 2);
			y2a = yc;
			x1b = 0;
			y1b = h - (y2a - y1a);
			x2b = std::min(w, x2a - x1a);
			y2b = h;
		}
		else if (i == 2) {  // bottom left
			x1a = std::max(xc - w, 0);
			y1a = yc;
			x2a = xc;
			y2a = std::min(s * 2, yc + h);
			x1b = w - (x2a - x1a);
			y1b = 0;
			x2b = w;
			y2b = std::min(y2a - y1a, h);
		}
		else if (i == 3) {  // bottom right
			x1a = xc;
			y1a = yc;
			x2a = std::min(xc + w, s * 2);
			y2a = std::min(s * 2, yc + h);
			x1b = 0;
			y1b = 0;
			x2b = std::min(w, x2a - x1a);
			y2b = std::min(y2a - y1a, h);
		}
		
		// 拷贝数据至目标roi区域
		cv::Mat roi = img4(cv::Rect(x1a, y1a, x2a - x1a, y2a - y1a));
		img(cv::Rect(x1b, y1b, x2b - x1b, y2b - y1b)).copyTo(roi);

		auto padw = x1a - x1b;
		auto padh = y1a - y1b;

		// load segments and get bbox define
		std::vector<std::vector<float>> bboxs;
		std::vector<std::vector<float>> lbs_segments;
		read_segment_labels(label_files[img_idx[i]], bboxs, lbs_segments);

		int labels_num = bboxs.size();
		torch::Tensor label_tensor = torch::zeros({ labels_num, 5 }).to(torch::kFloat32);
		if (label_tensor.sizes().size() == 1 || labels_num == 0)
		{
			LOG(WARNING) << "read boxs number 0, file: " << label_files[img_idx[i]];
		}

		for (int j = 0; j < labels_num; j++)
		{
			auto seg_tmp = torch::tensor(lbs_segments[j]).view({-1, 2});
			seg_tmp = xyn2xy(seg_tmp, w, h, padw, padh);
			segments.push_back(seg_tmp);
			label_tensor[j].index_put_({ torch::indexing::Slice() }, torch::tensor(bboxs[j]));
		}
		auto select_xyhw = label_tensor.index({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) });
		auto converted = xywhn2xyxy(select_xyhw, w, h, padw, padh);
		label_tensor.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) }, converted);
		labels.emplace_back(label_tensor);
		//std::cout << "     read " << i << " labels: " << labels[i].sizes() << std::endl;
	}

	// Concat labels
	for(int i = 0 ; i < labels.size(); i++)
	{
		if(labels[i].size(0) == 0)
			std::cout << ColorString("labels is empty: ", "R") << " labels4: " << i << " label: " << labels[i].sizes() << std::endl;
	}
	labels4 = torch::cat(labels, 0);
	labels4.clamp_(0, 2 * img_size);
	for(auto seg_tmp : segments)
	{
		// ???添加筛选处理等
		segments4.push_back(seg_tmp);
	}
	//std::cout << "labels4: " << labels4.sizes() << std::endl;

	auto [aug_img4, aug_labels, aug_segment] = random_perspective(img4, labels4, 
			segments4,
			int(std::get<float>(hyp["degrees"])), std::get<float>(hyp["translate"]),
			std::get<float>(hyp["scale"]), int(std::get<float>(hyp["shear"])),
			std::get<float>(hyp["perspective"]), mosaic_border);

	return std::make_tuple(aug_img4, aug_labels, aug_segment);
}


std::tuple<Dataloader_Custom, int> create_dataloader(
						const std::string& path,
                        int imgsz,
						int nc,
                        int batch_size,
                        int stride,
                        VariantConfigs& opt,
                        VariantConfigs& hyp,
                        bool augment,
                        float pad, 
						bool is_val,
						bool is_segment,
						bool is_overlap,
						int downsample_ratio)
{
	int gs = stride;
	bool rect = is_val ? true : std::get<bool>(opt["rect"]);
	bool quad = std::get<bool>(opt["quad"]);
	bool single_cls = nc == 1 || std::get<bool>(opt["single_cls"]);	
	auto datasets = LoadImagesAndLabels(path, 
										hyp,											
										imgsz, 
										augment, 
										rect, 
										single_cls, 
										gs, 
										pad,
										is_segment,
										is_overlap, 
										downsample_ratio).map(CustomCollate());
    int num_images = *(datasets.size());

	auto dataloader_options = torch::data::DataLoaderOptions().batch_size(batch_size).workers(std::get<int>(opt["workers"]));
#if 1	
	Dataloader_Custom dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                std::move(datasets), dataloader_options);
	return std::make_tuple(std::move(dataloader), num_images);
#else
	torch::data::samplers::SequentialSampler sampler(num_images);
	//torch::data::samplers::RandomSampler sampler(num_images);
	Dataloader_Custom dataloader = std::make_shared<Dataloader_CutomeType>(std::move(datasets), std::move(sampler), dataloader_options);
	return {dataloader, num_images};
#endif	
}


