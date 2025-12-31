#include "BaseData.h"

#include "yaml_load.h"
#include "utils.h"

#include "ops.h"

#include <filesystem>
#include <map>
#include <fstream>

namespace fs = std::filesystem;

std::string GET_VDK(VDK key)
{
std::unordered_map<VDK, std::string> dict_data = {
		{VDK::FILENAME, 		"filename"},		// std::vector<std::string>
		{VDK::INDEX, 			"index"},			// int
		{VDK::SRC_SIZE, 		"src_size"},		// std::vector<std::tuple<float, float>>
		{VDK::IMAGE, 			"image"},			// cv::Mat
		{VDK::IMG, 				"img"},				// torch::Tensor
		{VDK::CLS, 				"cls"},				// torch::Tensor
		{VDK::BBOXES, 			"bboxes"},			// torch::Tensor
		{VDK::BATCH_IDX, 		"batch_idx"},		// torch::Tensor
		{VDK::RATIO, 			"ratio"},			// std::vector<std::tuple<float, float>>
		{VDK::PAD, 				"pad"},				// std::vector<std::tuple<float, float>>
		{VDK::RESIZED, 			"resized"},			// std::vector<std::tuple<float, float>>
		{VDK::SEGMENTS, 		"segments"},		// torch::Tensor 
		{VDK::MOSAIC_BORDER, 	"mosaic_border"},	// std::vector<float>
		{VDK::MASK, 			"mask"},			// torch::Tensor
};
	return dict_data[key];
}


// ------------------------------------- BaseDataset Start -------------------------------------------------------

BaseDataset::BaseDataset(const std::string& path, VariantConfigs* pargs, int stride)
{
	m_path = path;
	m_stride = stride;
	m_pArgs = pargs;

	getSettings();

	std::tie(m_imagesDir, m_labelsDir) = getImageLabelLists(m_path, m_imgFiles, m_labelFiles);
	m_totalSamples = m_imgFiles.size();
}

void BaseDataset::getSettings()
{
	m_imgSize = getVariantValue<int>(m_pArgs, "imgsz", 640);
	m_bRect = getVariantValue<bool>(m_pArgs, "rect", false);
	m_bAugment = getVariantValue<bool>(m_pArgs, "augment", false);
	m_bSingleClass = getVariantValue<bool>(m_pArgs, "single_cls", false);
	auto mode = getVariantValue<std::string>(m_pArgs, "mode", "train");

	if (mode == "train")
		m_bAugment = true;

	m_bMosaic = m_bAugment && m_bRect == false;
}

void BaseDataset::initFinal()
{
	getSpecialSettins();
}


std::tuple<std::string, std::string> BaseDataset::getImageLabelLists(const std::string& img_path, 
	std::vector<std::string>& img_files, std::vector<std::string>& label_files)
{
	std::string images_dir;
	std::string labels_dir;
	std::string path = img_path;

	auto get_labelpath = [&](std::string _path) {
		auto fs_img_path = fs::path(_path);
		if (fs::is_directory(fs_img_path) && fs::exists(fs_img_path))
		{
			images_dir = path;
			std::string tmp = fs_img_path.filename().string();		//such as: "train2017"
			std::cout << "path: " << path << std::endl;
			std::cout << "data name: " << tmp << std::endl;
			std::cout << "data path: " << fs_img_path.parent_path().parent_path().string() << std::endl;

			labels_dir = fs_img_path.parent_path().parent_path().append("labels").append(tmp).string();

			if (false == fs::exists(std::filesystem::path(labels_dir)))
				LOG(ERROR) << "Labels folder: " << labels_dir << " not exists.";

		}
		};

	std::vector<std::string> list_imgs;
	auto parent_path = fs::path(path).parent_path().string();
	bool b_save_filter = true;

	auto read_files_from_txt = [&](const std::string& filename, std::vector<std::string>& lists) {
		std::ifstream file(filename);
		std::string line;
		if (file.is_open())
		{
			while (std::getline(file, line))
			{
				if (!line.empty() && line.back() == '\n')
					line.pop_back();

				if (!line.empty())
					lists.push_back(std::filesystem::path(parent_path).append(line).string());
			}
		}
		else {
			LOG(WARNING) << "open file: " << filename << " error!" << std::endl;
		}
		};

	if (fs::is_directory(fs::path(path)))
	{
		auto tmp_dirname = fs::path(path).stem().string();
		auto img_path_filter = fs::path(parent_path).append(tmp_dirname + ".filter");

		if (fs::exists(fs::path(img_path_filter)))	// 找到以前保存的filter文件，从中读取
		{
			std::cout << "check filter file: " << img_path_filter.string() << " Replace the original: " << path << std::endl;
			read_files_from_txt(img_path_filter.string(), list_imgs);
			images_dir = parent_path;
			labels_dir = fs::path(parent_path).parent_path().append("labels/" + tmp_dirname).string();
			path = img_path_filter.string();

			b_save_filter = false;
		}
		else
		{
			getAllImageInFolder(path, list_imgs);
			get_labelpath(path);
		}
	}
	else
	{	// 从txt文件读取，每行为图像:"./images/train2017/000000109622.jpg",根目录是coco
		auto img_path_filter = std::filesystem::path(path + ".filter");
		if (std::filesystem::exists(img_path_filter))
		{
			// 该文件是第一次数据集扫描生成的，已经整理过的数据，可以大大幅降低后续训练时载入时间
			std::cout << "check filter file: " << img_path_filter.string() << " Replace the original: " << path << std::endl;
			path = path + ".filter";
			b_save_filter = false;
		}
		//std::cout << "datasets image root: " << parent_path << std::endl;
		read_files_from_txt(path, list_imgs);
		//std::cout << "total read images name count: " << list_imgs.size() << std::endl;
		if (list_imgs.size())
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

	if (!std::filesystem::is_directory(std::filesystem::path(path))
		&& path.substr(path.length() - 7) == ".filter")
	{	// 读取以前的数据集，保存过的文件
		b_save_filter = false;

		for (int i = 0; i < list_imgs.size(); i++)
		{
			std::string item = list_imgs[i];
			std::string imgfile_stem = std::filesystem::path(item).stem().string();

			auto labelfile = std::filesystem::path(labels_dir).append(imgfile_stem + ".txt").string();
			if (i % 100 == 0) {
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
		auto check_fileexists = [&](int start_id, int end_id, const std::vector<std::string>& source_list, std::vector<std::string>& imgs, std::vector<std::string>& labels) {
			for (int i = start_id; i < end_id; i++)
			{
				std::string item = source_list[i];
				std::string imgfile_stem = std::filesystem::path(item).stem().string();
				auto labelfile = std::filesystem::path(labels_dir).append(imgfile_stem + ".txt").string();
				if (start_id == 0) {
					std::cout << "\x1b[2K\r" << "scan files: " << end_id << " | " << i << " ";
					std::cout << std::flush;
				}
				if (std::filesystem::exists(std::filesystem::path(item)) && std::filesystem::exists(std::filesystem::path(labelfile)))
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

		for (int threadid = 0; threadid < thread_count; threadid++)
		{
			int start_idx = threadid * temp_steps;
			int end_idx = std::min(int(list_imgs.size()), (start_idx + temp_steps));
			thread_filter[threadid] = std::thread(check_fileexists, start_idx, end_idx, std::ref(list_imgs),
				std::ref(temp_imgs[threadid]), std::ref(temp_labels[threadid]));
		}
		for (int threadid = 0; threadid < thread_count; threadid++)
			thread_filter[threadid].join();

		for (int v_id = 0; v_id < thread_count; v_id++)
		{
			for (int i = 0; i < temp_imgs[v_id].size(); i++)
			{
				img_files.emplace_back(temp_imgs[v_id][i]);
				label_files.emplace_back(temp_labels[v_id][i]);
			}
		}
		// std::cout << "total images: " << img_files.size() << std::endl;
	}
	if (b_save_filter)
	{	// 保存至原文件目录下
		auto save_path = path + ".filter";
		std::ofstream ofs(save_path);
		if (ofs.is_open())
		{
			for (int i = 0; i < img_files.size(); i++)
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

std::vector<std::vector<float>> BaseDataset::readLabelsFromFile(const std::string& filename)
{
	std::vector<std::vector<float>> result;
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "can't open file: " << filename << std::endl;
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

torch::Tensor createOneSegment(std::vector<std::vector<float>> segments, int numDimNorSize = 500)
{
	int nl = segments.size();	// number of total objects
	std::vector<torch::Tensor> vt_segments;
	for (int i = 0; i < nl; i++)
	{
		std::vector<float> segment = segments[i];
		int n_pt = segment.size() / 2;		// 有多少个线段
		int n_remainder = numDimNorSize % n_pt;
		int resample_size = (numDimNorSize - n_remainder) / n_pt;
		segment.emplace_back(segment[0]);
		segment.emplace_back(segment[1]);	// 最后添加第一个，方便闭合采集
		
		std::vector<float> new_segments;
		new_segments.reserve(numDimNorSize);

		for (int j = 0; j < n_pt; j++)
		{
			float x1 = segment[j * 2];
			float y1 = segment[j * 2 + 1];
			float x2 = segment[(j + 1) * 2];
			float y2 = segment[(j + 1) * 2 + 1];

			int one_steps = (j == n_pt - 1) ? float(resample_size) : float(resample_size + n_remainder);
			float d_x = (x2 - x1) / float(one_steps);
			float d_y = (y2 - y1) / float(one_steps);

			for (int k = 0; k < one_steps; k++)
			{
				new_segments.push_back(x1 + d_x * k);
				new_segments.push_back(y1 + d_y * k);
			}
		}
		auto s = torch::tensor(new_segments).view({ -1, 2 });
		vt_segments.emplace_back(s);
	}

	torch::Tensor t_segment = torch::stack(vt_segments, 0);
	return t_segment;
}

VariantData BaseDataset::transLabelsToBbox(std::vector<std::vector<float>>& labels, VariantData& data)
{
	std::vector<float> cls;
	std::vector<std::vector<float>> bbox;
	VariantData rets;
	if (labels.size() == 0)
	{
		LOG(ERROR) << "input labels size == 0";
		return rets;
	}

	if (labels[0].size() == 5)	// bboxes 
	{
		for (auto oneitem : labels)
		{
			if (oneitem.size() == 5)
			{
				cls.push_back(oneitem.front());
				bbox.emplace_back(oneitem.begin() + 1, oneitem.end());
			}
			else
			{
				LOG(ERROR) << "labels not same type." << oneitem.size() << " : 5";
			}
		}
	}
	else
	{
		std::vector<std::vector<float>> segments;
		for (auto oneitem : labels)
		{
			cls.push_back(oneitem[0]);
			float minx, miny, maxx, maxy;
			for (int i = 1; i < oneitem.size(); i+=2)
			{
				float x = oneitem[i];
				float y = oneitem[i + 1];
				if (i == 1)
				{
					minx = x;
					miny = y;
					maxx = x;
					maxy = y;
				}
				minx = std::min(minx, x);
				miny = std::min(miny, y);
				maxx = std::max(maxx, x);
				maxy = std::max(maxy, y);

			}
			float x_c = (minx + maxx) / 2.0f;
			float y_c = (miny + maxy) / 2.0f;
			float w = maxx - minx;
			float h = maxy - miny;
			bbox.emplace_back(std::vector<float>({ x_c, y_c, w, h }));
			segments.emplace_back(oneitem.begin() + 1, oneitem.end());
		}
		torch::Tensor t_segments = createOneSegment(segments);	// 通过得采样，统一为[n, M = 500, 2]
		rets[GET_VDK(VDK::SEGMENTS)] = t_segments;
	}

	torch::Tensor t_bbox = torch::zeros({int(bbox.size()), 4}).to(torch::kFloat32);
	for(int i = 0; i < bbox.size(); i++)
		t_bbox[i].index_put_({torch::indexing::Slice() }, torch::tensor(bbox[i]));
	
	torch::Tensor t_cls = torch::zeros({int(cls.size())}).to(torch::kFloat32);
	t_cls.index_put_({torch::indexing::Slice()}, torch::tensor(cls));
	t_cls = t_cls.view({-1, 1});

	rets[GET_VDK(VDK::CLS)] = t_cls;
	rets[GET_VDK(VDK::BBOXES)] = t_bbox;

	return rets;
}

VariantData BaseDataset::getImageAndLabel(size_t index)
{
	VariantData data;

	data[GET_VDK(VDK::FILENAME)] = std::vector<std::string>({ m_imgFiles[index] });
	data[GET_VDK(VDK::INDEX)] = int(index);
	auto [image, src_size, resized] = loadImage(index);
	data[GET_VDK(VDK::IMAGE)] = image;

	/*
	* 无pad，坐标体系等比例放缩，暂时不考虑原尺寸
	*/
	data[GET_VDK(VDK::SRC_SIZE)] = std::vector<std::tuple<float, float>>({ resized });

	auto labels = getLabel(index);
	auto rets = transLabelsToBbox(labels, data);

	data[GET_VDK(VDK::BBOXES)] = std::get<torch::Tensor>(rets[GET_VDK(VDK::BBOXES)]);
	data[GET_VDK(VDK::CLS)] = std::get<torch::Tensor>(rets[GET_VDK(VDK::CLS)]);
	if(rets.count(GET_VDK(VDK::SEGMENTS)))
		data[GET_VDK(VDK::SEGMENTS)] = std::get<torch::Tensor>(rets[GET_VDK(VDK::SEGMENTS)]);

	data["xyxy"] = int(-1);	// xywh > 0 , xyxy < 0, xyxy时，坐标是以data["image"]的坐标为基准

	return data;
}

std::tuple<cv::Mat, std::tuple<float, float>, std::tuple<float, float>> BaseDataset::loadImage(size_t index)
{
	cv::Mat img = cv::imread(m_imgFiles[index]);
	if (img.empty())
		LOG(ERROR) << "load image: " << m_imgFiles[index] << " failed.";

	int w0 = float(img.size().width);
	int h0 = float(img.size().height);
	int w = w0;
	int h = h0;
	if (m_bRect)	// 长边对齐imgsz
	{
		float ratio = float(m_imgSize) / float(std::max(w0, h0));
		w = std::min(m_imgSize, int(float(w) * ratio));
		h = std::min(m_imgSize, int(float(h) * ratio));
	}
	else
	{	// strech to [m_imgSize, m_imgSize]
		w = m_imgSize;
		h = m_imgSize;
	}
	cv::resize(img, img, cv::Size(w, h), 0.0, 0.0, cv::INTER_LINEAR);

	return { img, {float(w0), float(h0)}, {float(w), float(h)} };
}

std::vector<std::vector<float>> BaseDataset::getLabel(size_t index)
{
	return readLabelsFromFile(m_labelFiles[index]);
}


// ------------------------------------- End BaseDataset  -------------------------------------------------------


// ------------------------------------- BaseDataloader Start -------------------------------------------------------

BaseDataloader::BaseDataloader(const std::string& root_path, VariantConfigs& args_input, int stride, bool is_val)
{
	m_stride = stride;
	m_isVal = is_val;
	m_args = args_input;

	getDataSets(root_path);
}

int BaseDataloader::getNumberOfBatch()
{
	if (m_totalNumber == 0 || m_batchSize == 0) return 0;

	int remainder = m_totalNumber % m_batchSize;

	int tmp = (m_totalNumber - remainder) / m_batchSize;

	return remainder > 0 ? tmp + 1 : tmp;
}

void BaseDataloader::getDataSets(const std::string root)
{
	auto data_filepath = fs::path(root).append(std::get<std::string>(m_args["data"]));
	if (false == fs::exists(data_filepath))
	{
		LOG(ERROR) << "datasets yaml file: " << data_filepath.string() << " not exists.";
		return;
	}

	std::string train_path;
	std::string val_path;
	readDataYaml(data_filepath.string(), train_path, val_path, m_names);
	m_datasetsPath = m_isVal ? val_path : train_path;
	m_datasetsPath = fs::path(root).append(m_datasetsPath).string();

	m_batchSize = getVariantValue<int>(&m_args, "batch", 16);
	if (m_isVal)
		m_batchSize *= 2;

	m_args["mode"] = m_isVal ? "val" : "train";
	std::cout << "dataloader isVal: " << (m_isVal ? "val" : "train") << std::endl;
}

void BaseDataloader::initDatas()
{
	createDataloader();
}

void BaseDataloader::closeMosaic(bool bFlag)
{
	m_args["mosaic"] = bFlag ? 0.f : 1.f;
}


// ------------------------------------ - End BaseDataloader------------------------------------------------------ -

void BaseTransform::updateLabels(VariantData& label, int w, int h, int padw, int padh)
{
	if (label.count(GET_VDK(VDK::BBOXES)) > 0)
	{
		torch::Tensor bbox = std::get<torch::Tensor>(label[GET_VDK(VDK::BBOXES)]);
		label[GET_VDK(VDK::BBOXES)] = xywhn2xyxy(bbox, w, h, padw, padh);
	}

	if (label.count(GET_VDK(VDK::SEGMENTS)) > 0)
	{
		torch::Tensor segments = std::get<torch::Tensor>(label[GET_VDK(VDK::SEGMENTS)]);
		segments.index({"...", 0}).mul_(w).add_(padw);
		segments.index({"...", 1}).mul_(h).add_(padh);
	}
}

BaseTransformArray::BaseTransformArray(VariantConfigs* args, int imgsz, int dataset_size, bool augment)
	: m_pArgs(args), m_imageSize(imgsz), m_datasetSize(dataset_size), m_bAugment(augment)
{
}

void BaseTransformArray::initAllTrans()
{
	buildTranforms(m_bAugment);
}

void BaseTransformArray::doTransforms(BaseDataset* pdataset, size_t index, VariantData& pData)
{
	for(int i = 0; i < m_trans.size(); i++)
	{
		m_trans[i]->doTransform(pdataset, index, pData);
	}
}
