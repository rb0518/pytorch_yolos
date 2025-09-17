#include <filesystem>
#include <random>
#include <algorithm> // std::shuffle
#include <fstream>
#include <thread>

#include "datasets.h"
#include "utils.h"

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

inline torch::Tensor box_candidates(const torch::Tensor& box1,
	const torch::Tensor& box2,
	float wh_thr = 2.0f,
	float ar_thr = 20.0f,
	float area_thr = 0.1f,
	float eps = 1e-16f) 
{
	auto w1 = box1[2] - box1[0];
	auto h1 = box1[3] - box1[1];
	auto w2 = box2[2] - box2[0];
	auto h2 = box2[3] - box2[1];

	auto ar1 = w2 / (h2 + eps);
	auto ar2 = h2 / (w2 + eps);
	auto ar = torch::maximum(ar1, ar2);

	auto area_ratio = (w2 * h2) / (w1 * h1 + eps);

	return (w2 > wh_thr) &
		(h2 > wh_thr) &
		(area_ratio > area_thr) &
		(ar < ar_thr);
}


torch::Tensor transform_labels(const torch::Tensor& targets,
	const torch::Tensor& M,
	int width, int height,
	bool perspective,
	bool use_segments,
	float s = 1.0f)
{
	use_segments = false;	// ???

	int n = targets.size(0);
	if (n == 0) return targets.clone();

	auto xy = torch::ones({ n * 4, 3 }, torch::kFloat32);

	// 重构数据组织 (x1y1, x2y2, x1y2, x2y1)
	auto target_coords = targets.index({ torch::indexing::Slice(),
									  torch::tensor({1, 2, 3, 4, 1, 4, 3, 2}) });
	xy.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(0, 2) },
		target_coords.reshape({ n * 4, 2 }));

	// 坐标体系与图像同等transform 
	xy = xy.matmul(M.t());

	// 是否透视效果
	if (perspective) {
		auto xy_div = xy.index({ torch::indexing::Slice(), 2}).unsqueeze(1);
		xy = xy.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 2) }).div(xy_div);
	}
	else {
		xy = xy.index({ torch::indexing::Slice(), torch::indexing::Slice(0, 2) });
	}
	xy = xy.reshape({ n, 8 });
	// create new boxes
	auto x = xy.index({ torch::indexing::Slice(),
					  torch::tensor({0, 2, 4, 6}) });
	auto y = xy.index({ torch::indexing::Slice(),
					  torch::tensor({1, 3, 5, 7}) });

	auto x_min = std::get<0>(x.min(1));
	auto y_min = std::get<0>(y.min(1));
	auto x_max = std::get<0>(x.max(1));
	auto y_max = std::get<0>(y.max(1));

	auto new_boxes = torch::stack({ x_min, y_min, x_max, y_max }, 1);

	//裁切
	new_boxes.index_put_({ torch::indexing::Slice(),
						 torch::tensor({0, 2}) },
		new_boxes.index({ torch::indexing::Slice(),
					   torch::tensor({0, 2}) }).clamp(0, width));
	new_boxes.index_put_({ torch::indexing::Slice(),
						 torch::tensor({1, 3}) },
		new_boxes.index({ torch::indexing::Slice(),
					   torch::tensor({1, 3}) }).clamp(0, height));

	// filter candidate
	auto box1 = targets.index({ torch::indexing::Slice(),
							 torch::indexing::Slice(1, 5) }).t() * s;
	auto box2 = new_boxes.t();
	float thr = use_segments ? 0.01f : 0.10f;
	auto keep = box_candidates(box1, box2, thr);

	auto result = targets.index({ keep });
	result.index_put_({ torch::indexing::Slice(),
					  torch::indexing::Slice(1, 5) },
		new_boxes.index({ keep }));

	return result;
}

void augment_hsv(cv::Mat& image, float hgain = 0.5, float sgain = 0.5, float vgain = 0.5)
{
	// 生成随机增益 [-1,1]*gain + 1
	float r[3] = {
		random_uniform(-1.0f, 1.0f) * hgain + 1.0f,
		random_uniform(-1.0f, 1.0f) * sgain + 1.0f,
		random_uniform(-1.0f, 1.0f) * vgain + 1.0f
	};

	// 转换到HSV空间
	cv::Mat hsv;
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

	// 分离通道
	std::vector<cv::Mat> channels;
	cv::split(hsv, channels);

	// 创建LUT
	cv::Mat lut_hue(1, 256, CV_8U);
	cv::Mat lut_sat(1, 256, CV_8U);
	cv::Mat lut_val(1, 256, CV_8U);

	for (int i = 0; i < 256; ++i) {
		lut_hue.at<uchar>(i) = static_cast<uchar>(i * r[0]) % 180;
		lut_sat.at<uchar>(i) = cv::saturate_cast<uchar>(i * r[1]);
		lut_val.at<uchar>(i) = cv::saturate_cast<uchar>(i * r[2]);
	}

	// 应用LUT
	cv::LUT(channels[0], lut_hue, channels[0]);
	cv::LUT(channels[1], lut_sat, channels[1]);
	cv::LUT(channels[2], lut_val, channels[2]);

	// 合并通道并转换回BGR
	cv::merge(channels, hsv);
	cv::cvtColor(hsv, image, cv::COLOR_HSV2BGR);
}


std::tuple<cv::Mat, std::vector<float>, std::vector<float>>  letterbox(cv::Mat img,
	std::pair<int, int> new_shape = { 640, 640 },
	cv::Scalar color = cv::Scalar(114, 114, 114),
	bool auto_mode = true,
	bool scaleFill = false,
	bool scaleup = true,
	int stride = 32) 
{
	cv::Mat img_out = cv::Mat(new_shape.first, new_shape.second, CV_8UC3, cv::Scalar(114,114,114));
	std::vector<float> ratio = {1.f, 1.f};  // ratio [w, h]
	std::vector<float> pad ={0.f, 0.f};

	// load_image时已经根据比例放缩到了单边与imgsz一致了，暂时不考虑填充的模式
	auto h = img.rows;
	auto w = img.cols;

	auto new_h = h;
	auto new_w = w;

	if(scaleFill)	// stretch，将原图记放缩到与输出图像宽高一致，无边缘填充
	{
		pad = {0.f, 0.f};
		ratio[0] = float(new_shape.first) / float(w);
		ratio[1] = float(new_shape.first) / float(h);
		//std::cout << "letter stretch: " << ratio[0] << " " << ratio[1] << std::endl;
		new_w = new_shape.first;
		new_h = new_shape.first;
	}
	else
	{
		auto w_r = float(new_shape.first) / float(w);
		auto h_r = float(new_shape.first) / float(h);
		//std::cout << "wr " << w_r << " " << h_r << std::endl;
		auto min_ratio = std::min(w_r, h_r);
		ratio[0] = min_ratio;
		ratio[1] = min_ratio;
		new_w = std::min(new_shape.first, int(float(w) * ratio[0]));
		new_h = std::min(new_shape.first, int(float(h) * ratio[1]));
		pad[0] = (new_shape.first - new_w) / 2;
		pad[1] = (new_shape.first - new_h) / 2;
		//std::cout << "false, new_w " << new_w <<" " << new_h << " pad " << pad[0] << " " << pad[1] << std::endl;
	}
	
	cv::Mat new_img;
	cv::resize(img, new_img, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);

	cv::Mat roi = img_out(cv::Rect(int(pad[0]), int(pad[1]), new_w, new_h));
	new_img.copyTo(roi);

	return std::make_tuple(img_out, ratio, pad );
}
#if 0
std::pair<torch::data::DataLoaderBase<LoadImagesAndLabels>, LoadImagesAndLabels>  create_dataloader(const std::string& path,
                        int imgsz,
                        int batch_size,
                        int stride,
                        VariantConfigs& opt,
                        VariantConfigs& hyp,
                        bool augment /*= false*/,
                        bool cache /*= false*/,
                        float pad /*= 0.0f*/,
                        bool rect /*= false*/,
                        int rank /*= -1*/,
                        int world_size /*= 1*/,
                        int workers /*= 8*/,
                        bool image_weights /*= false*/,
                        bool quad /*= false*/,
                        const std::string& prefix /*= ""*/) 
{          
    // 分布式�?�理逻辑     
    if (rank != -1) 
    {         // 分布式同步逻辑实现...     
    }      
    // 创建数据�?     
	bool single_cls = std::get<bool>(opt["single_cls"]);
    auto dataset = LoadImagesAndLabels(path, imgsz, batch_size,
        augment, hyp, rect, cache,
        std::get<bool>(opt["single_cls"]), stride, pad,
        image_weights, prefix);      
    // 调整batch大小     
    batch_size = std::min(batch_size, static_cast<int>(dataset.size().value()));      
    // 计算worker数量     
    int num_workers = std::min({static_cast<int>(std::thread::hardware_concurrency()) / world_size,  
        batch_size > 1 ? batch_size : 0, workers});      
        // 创建采样�?     
    torch::data::samplers::DistributedSampler ds_sampler(dataset.size().value(), rank, world_size);      
        // 创建数据加载�?     
	auto loader_options = torch::data::DataLoaderOptions().batch_size(batch_size)
                                                                    .workers(num_workers)
                                                                    .sampler(ds_sampler)
                                                                    .pin_memory(true)
																	.collate_fn(&LoadImagesAndLabels::collate_fn);
	if(quad)
		loader_options = loader_options.collate_fn(&LoadImagesAndLabels::collate_fn4);
	
    auto dataloader = torch::data::make_data_loader(std::move(dataset), loader_options);      
    return {std::move(*dataloader), dataset}; 
} 
#endif

LoadImagesAndLabels::LoadImagesAndLabels(std::string _path, VariantConfigs _hyp, int _img_size, int _batch_size, 
	bool _augment, bool _rect, bool _image_weights, bool _cache_images, bool _single_cls, int _stride, 
	float _pad, std::string _prefix)
	: img_size(_img_size), augment(_augment), image_weights(_image_weights), catch_images(_cache_images),
	single_cls(_single_cls), stride(_stride), pad(_pad), path(_path)
{
	rect = image_weights ? false : _rect;
	mosaic = augment == true && rect == false;
	//mosaic = false;
	int half_size = static_cast<int>(std::trunc(float(img_size) / 2.0));
	mosaic_border = { -half_size, -half_size };

	if(_hyp.size())
		for (auto& [k, v] : _hyp)	
			hyp[k] = v;

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
	auto parent_path = std::filesystem::path(this->path).parent_path().string();
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



	if (std::filesystem::is_directory(std::filesystem::path(this->path)))
	{
		auto tmp_dirname = std::filesystem::path(this->path).stem().string();
		auto img_path_filter = std::filesystem::path(parent_path).append(tmp_dirname + ".filter");

		if(std::filesystem::exists(std::filesystem::path(img_path_filter)))
		{
			std::cout << "check filter file: " << img_path_filter.string() <<" Replace the original: " << path << std::endl;	
			read_files_from_txt(img_path_filter, list_imgs);		
			images_dir = parent_path;
			labels_dir = std::filesystem::path(parent_path).parent_path().append("labels/" + tmp_dirname).string();
			//std::cout << "convert labels_dir: " << labels_dir << std::endl;
			path = img_path_filter;
			//std::cout << "new path: " << path << " sub: " << path.substr(this->path.length() - 7) << std::endl;

			b_save_filter = false;
		}
		else
		{
			listallfiles_withsuffixes(std::filesystem::path(path), list_imgs, img_format);
			get_labelpath(this->path);
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
		read_files_from_txt(this->path, list_imgs);
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

	if(!std::filesystem::is_directory(std::filesystem::path(this->path))
		&& path.substr(this->path.length() - 7) == ".filter")
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
		auto save_path = this->path+".filter";
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
	indices = random_queue(img_files.size());
}

torch::data::Example<> LoadImagesAndLabels::get(size_t index)
{
	auto indices_index = indices[index];
	auto need_mosaic = this->mosaic&& random_uniform() < std::get<float>(hyp["mosaic"]);

	cv::Mat img;
	torch::Tensor labels;
	torch::Tensor img_tensor;

	if (need_mosaic)
	{
		std::tie(img, labels) = load_mosaic(indices_index);
		//std::cout << "load mosaic first over..." << std::endl;
		if (random_uniform() < std::get<float>(hyp["mixup"]))
		{
			auto [img2, labels2] = load_mosaic(int(random_uniform(0, indices.size() - 1)));
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
			//std::cout << "load mosaic scend over..." << std::endl;
		}
		//std::cout << "load mosaic over..." << std::endl;
	}
	else
	{
		std::vector<int> hw0, hw;
		std::tie(img, hw0, hw) = load_image(indices_index);
		//std::cout << "load_mage: " << img.cols << " " << img.rows << std::endl;
		//cv::imshow("load image", img);

		std::vector<float> ratio, pad;
		std::tie(img, ratio, pad)= letterbox(img, std::make_pair(img_size, img_size),
			cv::Scalar(114, 114, 114), false, false, augment, stride);

		std::vector<std::vector<float>> shapes;
		shapes.emplace_back(std::vector<float>(hw0[0], hw0[1]));
		shapes.emplace_back(ratio);
		shapes.emplace_back(pad);

		// 调入labels, ???暂时不考虑segment
		std::vector<BBox_xyhw> bboxs;
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
			labels[j][1] = bboxs[j].x;
			labels[j][2] = bboxs[j].y;
			labels[j][3] = bboxs[j].w;
			labels[j][4] = bboxs[j].h;
			labels[j][0] = bboxs[j].cls_id;
		}
		auto w = ratio[0] * hw[1];
		auto h = ratio[1] * hw[0];
		//std::cout << "w " << w << " h " << h << " pad " << pad[0] << " " << pad[1] << std::endl;
		//std::cout << "labels: " << labels.sizes() << std::endl;
		auto select_xyhw = labels.index({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) });
		auto converted = xywhn2xyxy(select_xyhw, ratio[0] * hw[1], ratio[1] * hw[0], pad[0], pad[1]);
		labels.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) }, converted);

		// cv::Mat img_clone = img.clone();
		// for (int j = 0; j < labels.size(0); j++)
		// {
		// 	auto x1 = int(labels[j][1].item().toInt());
		// 	auto y1 = int(labels[j][2].item().toFloat());
		// 	auto x2 = int(labels[j][3].item().toFloat());
		// 	auto y2 = int(labels[j][4].item().toFloat());

		// 	cv::rectangle(img_clone, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)),
		// 		Colors::getInstance().get_color_scalar(int(labels[j][1].item().toInt())) , 2);
		// }
		// cv::imshow("letterbox image", img_clone);
		// cv::waitKey(); 
	}
	//std::cout << "load image over..." << std::endl;

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

	
	// 测试代码，查看图像
	// cv::Mat img_clone = img.clone();
	// //std::cout << "labels count: " << labels.size(0) << std::endl;
	// for (int j = 0; j < labels.size(0); j++)
	// {
	// 	auto x = labels_out[j][2].item().toFloat();
	// 	auto y = labels_out[j][3].item().toFloat();
	// 	auto w = labels_out[j][4].item().toFloat();
	// 	auto h = labels_out[j][5].item().toFloat();

	// 	x *= float(img_clone.cols);
	// 	y *= float(img_clone.rows);
	// 	w *= float(img_clone.cols);
	// 	h *= float(img_clone.rows);
	// 	//std::cout << j << " xywh " << x << " " << y << " - " << w <<" " << h << std::endl;

	// 	auto x1 = int(x - w /2.f);
	// 	auto y1 = int(y - h /2.f);
	// 	auto x2 = int(x + w /2.f);
	// 	auto y2 = int(y + h /2.f);
	// 	//std::cout << j << " xyxy " << x1 << " " << y1 << " - " << x2 <<" " << y2 << std::endl;

	// 	cv::rectangle(img_clone, cv::Point(x1, y1), cv::Point(x2, y2),
	// 		Colors::getInstance().get_color_scalar(int(labels_out[j][1].item().toInt())), 2);
	// }

	// cv::imshow("InputImage", img_clone);
	// cv::waitKey(); 

	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);	// BGR==>RGB
	img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte)
                                   .permute({2, 0, 1});	// HWC ==> CHW
	return {img_tensor.clone(), labels_out.clone()};
}

// 调入数据
std::tuple<cv::Mat, std::vector<int>, std::vector<int>> LoadImagesAndLabels::load_image(int index)
{	
	cv::Mat img = cv::imread(img_files[index]);
	if (img.empty())
		LOG(ERROR) << "load image: " << img_files[index] << " failed.";

	int h0 = img.rows;
	int w0 = img.cols;

	float r = float(img_size) / float(std::max(h0, w0));
	auto interpolation = r < 1.0 && augment ? cv::INTER_AREA : cv::INTER_LINEAR;
	cv::resize(img, img, cv::Size(int(w0 * r), int(h0 * r)), 0.0, 0.0, interpolation);
	std::vector<int> hw0 = { h0, w0 };				// image size数据格式传递统一为[height, width]
	std::vector<int> hw = { img.rows, img.cols };	
	return std::make_tuple(img, hw0, hw);
}

std::tuple<cv::Mat, torch::Tensor> LoadImagesAndLabels::load_mosaic(int index)
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
		auto [img, hw0, hw] = load_image(img_idx[i]);
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
		std::vector<BBox_xyhw> bboxs;
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
			label_tensor[j][0] = bboxs[j].cls_id;
			label_tensor[j][1] = bboxs[j].x;
			label_tensor[j][2] = bboxs[j].y;
			label_tensor[j][3] = bboxs[j].w;
			label_tensor[j][4] = bboxs[j].h;
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

	// test show the mosaic load is right.
	// cv::Mat img4_clone = img4.clone();
	// for(int j = 0; j < labels4.size(0); j++)
	// {
	// 	auto x1 = int(labels4[j][2].item().toInt());
	// 	auto y1 = int(labels4[j][3].item().toFloat());
	// 	auto x2 = int(labels4[j][4].item().toFloat());
	// 	auto y2 = int(labels4[j][5].item().toFloat());

	// 	cv::rectangle(img4_clone, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)),
	// 		cv::Scalar(0, 0, 220), 2);
	// }

	//cv::imshow("img4", img4_clone);
	//cv::waitKey();
	
	// Augment ��ǿ
	//	int degrees, float translate, float scale,
	//  int shear, float perspective, std::vector<int> border)
	//std::cout << "start random_perspective..." << std::endl;

	auto [aug_img4, aug_labels] = random_perspective(img4, labels4,
			int(std::get<float>(hyp["degrees"])), std::get<float>(hyp["translate"]),
			std::get<float>(hyp["scale"]), int(std::get<float>(hyp["shear"])),
			std::get<float>(hyp["perspective"]), mosaic_border);

	return std::make_tuple(aug_img4, aug_labels);
}

void LoadImagesAndLabels::load_xyhw_labels(const std::string& filename, std::vector<BBox_xyhw>& bboxs)
{
	std::ifstream file(filename);
	if (!file.is_open())
	{
		LOG(WARNING) << "open file " << filename << "fail.";
		return;
	}

	std::string line;
	while (std::getline(file, line))
	{
		BBox_xyhw tmp;
		std::stringstream ss(line);
		if (ss >> tmp.cls_id >> tmp.x >> tmp.y >> tmp.w >> tmp.h)
		{
			bboxs.push_back(tmp);
			//            std::cout << line << std::endl;
			//            std::cout << tmp.cls_id << ":" << tmp.x << " " << tmp.y <<" - " << tmp.w << " " << tmp.h << std::endl;
		}
		else
		{
			LOG(WARNING) << "read label data error!";
		}
	}
	file.close();
}

std::tuple<cv::Mat, torch::Tensor> LoadImagesAndLabels::random_perspective(cv::Mat img, torch::Tensor targets,
	int degrees, float translate, float scale,
	int shear, float perspective, std::vector<int> border)
{
	// targets = [cls, xyxy]
	//std::cout << "before random_perspective: [" << img.cols << " " << img.rows << "]" << std::endl;
	auto height = img.rows + border[0] * 2;
	auto width = img.cols + border[1] * 2;
	//std::cout << "           height width: [" << height << " " << width << "]" << std::endl;

	// Center
	auto C = torch::eye(3);
	C[0][2] = -(img.cols / 2);	// x translation(pixels)
	C[1][2] = -(img.rows / 2);	// y translation(pixels)
	//std::cout << "C: " << C << std::endl;
	//
	auto P = torch::eye(3);
	P[2][0] = random_uniform(-perspective, perspective); // x perspective (about y)
	P[2][1] = random_uniform(-perspective, perspective); // y perspective (about x)
	//std::cout << "P: " << P << std::endl;
	// Rotation and Scale
	auto R = torch::eye(3);
	auto a = random_uniform(-degrees, degrees);
	auto s = random_uniform(1 - scale, 1 + scale);

	float rad = a * M_PI / 180.0f;
	float cos_val = s * cos(rad);
	float sin_val = s * sin(rad);

	// ������?����ǰ����
	R.index_put_({ 0, 0 }, cos_val);
	R.index_put_({ 0, 1 }, -sin_val);
	R.index_put_({ 1, 0 }, sin_val);
	R.index_put_({ 1, 1 }, cos_val);
	//std::cout << "R: " << R << std::endl;
	// Shear
	auto S = torch::eye(3);
	// x shear (deg)
	S[0][1] = tan(random_uniform(-shear, shear) * M_PI / 180);
	// y shear(deg)
	S[1][0] = tan(random_uniform(-shear, shear) * M_PI / 180);
	//std::cout << "S: " << S << std::endl;
	// Translation
	auto T = torch::eye(3);
	T[0][2] = random_uniform(0.5 - translate, 0.5 + translate) * width;  // x translation(pixels)
	T[1][2] = random_uniform(0.5 - translate, 0.5 + translate) * height;  // y translation(pixels)
	//std::cout << "T: " << T << std::endl;
	// ������ϱ任����? (�ҳ�˳��)
	auto M = T.matmul(S).matmul(R).matmul(P).matmul(C);

	// ����Ƿ����?�任
	bool need_transform = border[0]!= 0 || border[1]!=0 ||
		(M != torch::eye(3, torch::kFloat32)).any().item<bool>();
	//std::cout << "need_transform : " << need_transform << std::endl;
	//std::cout << "M : " << M << std::endl;
	if (need_transform)
	{
		cv::Mat cv_M(3, 3, CV_32F);
		cv::Scalar border_value = { 114, 114, 114 };
		std::memcpy(cv_M.data, M.data_ptr(), M.numel() * sizeof(float));

		cv::Mat output;
		if (perspective) {
			cv::warpPerspective(img, output, cv_M, cv::Size(width, height),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
		}
		else {
			// ��ȡǰ������Ϊ����任����?
			cv::Mat affine_M = cv_M(cv::Rect(0, 0, 3, 2));
			cv::warpAffine(img, output, affine_M, cv::Size(width, height),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
		}
		img = output;
	}
	//std::cout << "after rand_perspective: [" << img.cols << " " << img.rows << "]" << std::endl;
	// Transform label coordinates
	targets = transform_labels(targets, M, width, height, perspective, false, s);

	// ת��ΪOpenCV����
	return { img, targets }; //std::make_tuple(img, targets);
}


