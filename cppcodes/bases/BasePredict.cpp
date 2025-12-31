#include <filesystem>

#include "BasePredict.h"

#include "utils.h"
#include "torch_utils.h"

#include "augmentations.h"

namespace fs = std::filesystem;

BasePredict::BasePredict(VariantConfigs& args_input, std::string root_dir)
{
	m_args = args_input;
	m_rootDir = root_dir;

	readSettings();
	initDevice();
	initDirs();
}

void BasePredict::setupPredict()
{
	readSpecialSettings();
	
	setupModel();
	setupSpecialObjects();
}

void BasePredict::doPredictTask()
{
	torch::NoGradGuard nograd;

	std::vector<std::string> test_files;
	getAllImageInFolder(m_sourceDir, test_files, true);
	std::cout << ColorString("start predict: ") << " total images " << test_files.size() << " in folder: " << m_sourceDir << std::endl;

	m_modelPtr->eval();	
	
	for (int i = 0; i < test_files.size(); i++)
	{
		cv::Mat s_mat;
		torch::Tensor s_tensor, preds, proto;
		std::tie(s_mat, s_tensor) = convertMatFileToTensor(test_files[i]);
		s_tensor = s_tensor.to(m_device);
		std::tie(preds, std::ignore, proto) = m_modelPtr->forward(s_tensor);

		auto rets = postprocess(preds);
		plotsResultToMat(s_mat, test_files[i], rets, proto);
	}
}

/*
# Predict settings -----------------------------------------------------------------------------------------------------
source: # (str, optional) source directory for images or videos
vid_stride: 1 # (int) video frame-rate stride
stream_buffer: False # (bool) buffer all streaming frames (True) or return the most recent frame (False)
visualize: False # (bool) visualize model features
augment: False # (bool) apply image augmentation to prediction sources
agnostic_nms: False # (bool) class-agnostic NMS
classes: # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
retina_masks: False # (bool) use high-resolution segmentation masks
embed: # (list[int], optional) return feature vectors/embeddings from given layers
*/
void BasePredict::readSettings()
{
	m_sourceDir = getVariantValue<std::string>(&m_args, "source", "");
	if (m_sourceDir == "" || false == fs::exists(fs::path(m_rootDir).append(m_sourceDir)))
	{
		std::cout << ColorString("source: ", "Error") << m_sourceDir << " wrong, use default value: data/images " << std::endl;
		m_sourceDir = "data/images";
	}
	m_sourceDir = fs::path(m_rootDir).append(m_sourceDir).string();
}

void BasePredict::initDevice()
{
	std::string device_name = getVariantValue<std::string>(&m_args, "device", "cpu");
	std::cout << "input device type: " << device_name << "\n";
	m_device = get_device(device_name);
	std::cout << ColorString("Device: ") << m_device.type() << std::endl;
}

void BasePredict::initDirs()
{
	if (getVariantValue<std::string>(&m_args, "project", "") == "")
		m_args["project"] = "runs/test";

	if (getVariantValue<std::string>(&m_args, "name", "") == "")
		m_args["name"] = "exp";

	std::tie(m_saveDir, std::ignore, std::ignore, std::ignore) = getSaveDirAndFilenames(m_args, m_rootDir, false);
	fs::create_directories(fs::path(m_saveDir));
	m_args["save_dir"] = fs::relative(fs::path(m_saveDir), fs::path(m_rootDir)).string();
}

void BasePredict::setupModel()
{
	std::string model_set = getVariantValue<std::string>(&m_args, "model", "");
	auto tmp_cfg_file_path = fs::path(m_rootDir).append(model_set);
	if (fs::exists(tmp_cfg_file_path) == false ||
		tmp_cfg_file_path.extension().string() != ".pt")
	{
		LOG(ERROR) << "model file not exists or is not a pt file: " << model_set;
		exit(-1);
	}
	auto cfg_file = tmp_cfg_file_path.string();
	m_modelPtr = createModel(cfg_file);
	if (m_modelPtr)
		loadWeights(cfg_file);
}

std::tuple<cv::Mat, torch::Tensor> BasePredict::convertMatFileToTensor(const std::string& filename)
{
	cv::Mat source_mat = cv::imread(filename);

	cv::Mat src_image;
	std::tie(src_image, std::ignore, std::ignore) = letterbox(src_image, std::make_tuple(m_imageSize, m_imageSize),
		cv::Scalar(114, 114, 114), false, m_bScaleFill, false, 32);	// scale_fill = true, 简化后续画图操作

	cv::Mat input_image;
	cv::cvtColor(src_image, input_image, cv::COLOR_BGR2RGB);

	input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);
	at::Tensor input_tensor = torch::from_blob(input_image.data,
		{ 1, input_image.rows, input_image.cols, 3 });    // ����batch_sizeά
	input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).contiguous();

	return { source_mat, input_tensor };
}
