#include "utils.h"

#include <filesystem>
#include <opencv2/dnn/dnn.hpp>  // nms要用到

#include <torch/script.h>       // load torchscript file
#include <regex>                // for std::regex_replace
#include <algorithm>
#include <fstream>
#include <random>
#include <algorithm> // std::shuffle
#include <chrono>

#include "plots.h"
#include "yaml_load.h"

int searchfiles_in_folder(const std::string& folder, const std::string& exttype, 
        std::vector<std::string>& lists)
{
    lists.clear();
    for(const auto& entry : std::filesystem::directory_iterator(folder))
    {
        if (entry.is_regular_file())
        {
            auto filename = entry.path().stem().string();
            auto fileext = entry.path().extension().string();
            if (fileext == exttype)
                lists.push_back(filename);
        }
    }
    return static_cast<int>(lists.size());
}

void view_jitmodel_info(torch::jit::Module module, int layer_id)
{
    std::stringstream ss;
    for (int i = 0; i <= layer_id; i++)
        ss << "  ";

    int ch_count = 0;
    for (const auto& submodule : module.named_modules())
    {
        if (ch_count != 0)
            view_jitmodel_info(submodule.value, layer_id + 1);
        else
            std::cout << ss.str() << submodule.name << std::endl;
    }
}


int LoadWeightFromJitScript(const std::string strScriptfile, torch::nn::Module& model, bool b_showinfo)
{
    bool load_is_ok = false;
    torch::Device device = model.parameters().begin()->device();

    torch::jit::script::Module jit_model;
    try {
        jit_model = torch::jit::load(strScriptfile);
        // 1、转存时已经将网上的yolov5s.pt从half-->float, 并指定在kCUDA上
        // 2、目前是用的trace，由于有可能有动态控制流，是否要用torch.jit.script，待测
        // 3、register_module注册子模块以保持参数名一致性
        jit_model.to(device);   // 与model的device保持一致
        jit_model.to(torch::kFloat);
        load_is_ok = true;
    }
    catch (const c10::Error& e)
    {
        std::cerr << e.what() << "\n";
        //std::exit(EXIT_FAILURE);
        return 0;
    }
    catch (...) {
        std::cerr << "torch::jit::load other exception \n";
        //std::exit(EXIT_FAILURE);
        return 0;
    }

    // 从pytorch的模型中加载预训练模型
    // 加载TorchScript模型

    // 提取参数到自定义模型
    // 将torch::jit_name_parameters_list重新组织为unordered_map，方便查找
    std::unordered_map<std::string, torch::Tensor> jit_params;
    for (const auto& param : jit_model.named_parameters())
    {
        jit_params[param.name] = param.value;
//        std::cout << "jit_model:  " << param.name << std::endl;
    }
    auto jit_params_count = jit_params.size();
    int n_params = 0;
    int count_changes = 0;
#ifdef _DEBUG_FOR_EXPORT_IMPORT_TENSORS_
    std::vector<std::string> match_layers;
    match_layers.push_back("model.23.");
#endif

#if 0
    int jit_model_count = 0;
    for (const auto& submodule : jit_model.named_modules())
    {
        std::cout << "Jit module: " << jit_model_count << std::endl;
        view_jitmodel_info(submodule.value, 0);
        jit_model_count++;
    }
#endif

    for (auto& param : model.named_parameters())
    {
        torch::NoGradGuard no_grad;   // 对模板进行操作时，不能是目标与源tensor一个有grad,一个无grad      
        std::string str_name = param.key();
//        std::cout << "model named_parameters: " << str_name << std::endl;
        //  "model-1.m-1.bn1.bias" ==> "model.1.m.1.bn1.bias
        str_name = std::regex_replace(str_name, std::regex("-"), ".");
#ifdef _DEBUG_FOR_EXPORT_IMPORT_TENSORS_        
        auto ret_str = findSubstringInStrings(str_name, match_layers);
        if (jit_params.find(str_name) != jit_params.end() && ret_str != "")
#else
        if (jit_params.find(str_name) != jit_params.end())
#endif
        {
            if (jit_params[str_name].sizes() == param.value().sizes())
            {
                // std::cout << "trans ok : " << str_name << std::endl;
                count_changes += 1;
                param.value().data().copy_(jit_params[str_name].data());
            }
            else if (b_showinfo)
            {   // 找到了，但尺寸不匹配，显示方便后续调试要检测目标尺寸与当前shape是否一致
                std::cout << str_name << ": " << std::endl;
                std::cout << "jit tensor " << jit_params[str_name].sizes() << " " << jit_params[str_name].dtype() << std::endl;
                std::cout << "model tensor " << param.value().sizes() << " " << param.value().dtype() << std::endl;
            }
        }
        else{
            std::cout <<"miss name: " << str_name << std::endl;
        }
        n_params += 1;
    }
    std::cout <<"Load jit parameters total: " << jit_params_count << " Model parameters total: " << n_params << " trans_count: " << count_changes << " paramters." << std::endl;
    return count_changes;
}

void init_torch_seek(int seed/* = 0*/)
{
    std::srand(seed);

    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);
    torch::cuda::manual_seed_all(seed);

    // ???
    // torch::globalContext().setDeterministicAlgorithms(true, true);
    // torch::globalContext().setDeterministicCuDNN(true);
    torch::globalContext().setDeterministicCuDNN(false);
    torch::globalContext().setDeterministicAlgorithms(false, false);
}

// 进程组ProcessGroup核心实现
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
// 通信相关基础功能
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/autograd/context/context.h>
// 太复杂了，暂时不用多卡或多机联合训练
void torch_distribute_zero_first(int local_rank,
    const std::shared_ptr<c10d::ProcessGroup>& pg)
{
    if (local_rank != -1 && local_rank != 0)
    {
        pg->barrier()->wait();
    }
    if (local_rank == 0)
        pg->barrier()->wait();
}

// Generate an incremental queue and randomly shuffle the order
std::vector<int> random_queue(int n) 
{
	std::vector<int> permutation(n);
	std::iota(permutation.begin(), permutation.end(), 0); // create list [1, 2, ..., n-1]

	// shuffle the order
	std::random_device rd;
	std::mt19937 g(rd()); 
	std::shuffle(permutation.begin(), permutation.end(), g); 

	return permutation;
}

float random_beta(float alpha, float beta) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::gamma_distribution<float> gamma_alpha(alpha, 1.0);
	std::gamma_distribution<float> gamma_beta(beta, 1.0);

	float x = gamma_alpha(gen);
	float y = gamma_beta(gen);
	return x / (x + y);
}


float random_uniform(float start/* = 0.0f*/, float end/* = 1.0f*/)
{
	std::random_device rd;  
	std::mt19937 gen(rd()); 

	std::uniform_real_distribution<> dis(start, end);

	return dis(gen);
}

std::tuple<bool, int, std::string, std::string> parse_yolo_config(const std::string& filename) 
{
    // 默认值设置
    int version_num;
    std::string size_type = "n";
    std::string task_type = "";

    // 正则表达式模式（支持yolov(n)('n/l/m/x'){-seg/obb/pose/cls}.yaml格式）
    std::regex pattern(R"(^yolo([v])?(\d+)([nlsmx])?(?:-(\w+))?\.yaml$)");
    std::smatch matches;
    bool is_valid = false;

    if (std::regex_match(filename, matches, pattern)) 
    {
        // for (int i = 0; i < matches.size(); i++)
        // {
        //     std::cout << " matches[" << i << "] " << matches[i].str() << "\n";
        // }

        is_valid = true;
        // version_num = matches[1].matched ? std::stoi(matches[1].str()) : 12;
        if (matches[2].matched) { // 必须带版本号
            version_num = std::stoi(matches[2].str());
        } else {
            is_valid = false;
            LOG(ERROR) << "filename not have yolo version: " << filename;
            return std::make_tuple(is_valid, version_num, size_type, task_type);
        }    

        size_type = matches[3].matched ? matches[2].str() : "";
        task_type = matches[4].matched ? matches[3].str() : "";
    }

    return std::make_tuple(is_valid, version_num, size_type, task_type);
}



// 2025-10-30 整理代码，调试工具，读取pytorch代码输出的Tensor文件
// 函数：读取二进制文件到内存缓冲区
std::vector<char> readBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open binary file " << filename << std::endl;
        return {};
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error: Failed to read the entire file " << filename << std::endl;
        return {};
    }
    
    file.close();
    return buffer;
}

torch::Tensor load_tensordata_from_file(std::string filename)
{
    std::vector<char> buffer = readBinaryFile(filename);

    torch::IValue x_pickle = torch::pickle_load(buffer);
    torch::Tensor x_clone = x_pickle.toTensor();
    std::cout << "read: " << filename << " x: " << x_clone.sizes() << std::endl;

    // 可以修改参数，引入目标Tensor，在这里做判定
    // if(x.sizes() == x_clone.sizes())
    // {
    //     std::cout << "sizes() equil, copy.\n";
    // }
    // else{
    //     std::cout << "sizes() error, return x\n";
    // }
    /*

    auto x_clone_sizes = x_clone.element_size() * x_clone.numel();
    std::cout << "load_tensordata_from_file, target size: " << x_clone.sizes() 
        << " total_size: " << x_clone_sizes << std::endl;
    std::cout << "load file byte count: " << buffer.size() << std::endl;

    if (x_clone_sizes == buffer.size())
    {
        std::cout << "copy buffer to tensor...";
        std::memcpy(x_clone.data_ptr(), buffer.data(), buffer.size());
        std::cout << "copy over..." << std::endl;
    }

    x_clone = x_clone.to(x.device());
    */
    return x_clone;
}

/*
*  load one txt file to memory
*/
bool readFileToMemory(const std::string& filePath, std::vector<unsigned char>& buffer) 
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }

    // allocate memory 
    file.seekg(0, std::ios::end);
    auto fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    buffer.clear();
    buffer.resize(fileSize);

    if (!file.read(reinterpret_cast<char*>(buffer.data()), fileSize)) {
        std::cerr << "Failed to read file: " << filePath << std::endl;
        return false;
    }

    return true;
}

torch::Tensor convertMemoryToTensor(std::vector<unsigned char>& buffer)
{
    return torch::tensor(buffer, c10::TensorOptions().dtype(torch::kByte));
}

void convertTensorToMemory(torch::Tensor x, std::vector<unsigned char>& buffer)
{
    auto t_size = x.element_size() * x.numel();
    buffer.clear();
    buffer.resize(t_size);
    std::memcpy(buffer.data(), x.data_ptr(), t_size);
}


