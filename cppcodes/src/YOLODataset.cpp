#include "YOLODataset.h"
#include "yaml-cpp/yaml.h"

#include <filesystem>
#include <opencv2/opencv.hpp>

YOLODataset::YOLODataset(const std::string &datacfg /*path to data/coco128.yaml*/,
                         const std::string &img_ext /*".jpg"*/,
                         int _width, int _height, bool _isTrain)
{
    if (!std::filesystem::exists(datacfg))
    {
        LOG(ERROR) << "Select " << datacfg << " no exists.";
        return;
    }
    isTrain = _isTrain;
    width = _width;
    height = _height;
    YAML::Node cfgs = YAML::LoadFile(datacfg);

    std::string data_root = cfgs["path"].as<std::string>();
    std::string imagedir = cfgs["train"].as<std::string>();
    
    if (false == _isTrain)
        imagedir = cfgs["val"].as<std::string>();

    // 不用直接赋值的原因是不同版本的coco.yaml中这里的定义格式有多种，写成循环好修改
    for (int i = 0; i < cfgs["names"].size(); i++)
    {
        class_names.push_back(cfgs["names"][i].as<std::string>());
    }

    // check
    if (class_names.size() != cfgs["nc"].as<int>())
    {
        LOG(WARNING) << "Read class names not equil the nc settings";
    }

    if (check_dirs(data_root, imagedir))
    {
        auto find_images_in_folder = [&]()
        {
            this->list_images.clear();
            //std::cout << this->images_dir << std::endl;
            for (const auto &entry : std::filesystem::directory_iterator(this->images_dir))
            {
                //std::cout << entry.path() << std::endl;
                if (entry.is_regular_file())
                {
                    auto filename = entry.path().stem().string();
                    auto fileext = entry.path().extension().string();
                    if (fileext == img_ext)
                        this->list_images.push_back(filename);
                }
            }
        };
        find_images_in_folder();
        //    int img_cout = searchfiles_in_folder(this->images_dir, img_ext, this->list_images);
        // 滤除labels不存在的文件，coco128中有000000000508.txt就找不到
        list_images.erase(std::remove_if(list_images.begin(), list_images.end(), [&](std::string itemname)
                                { 
                                    auto labelfilename = std::filesystem::path(labels_dir).append(itemname+".txt");
                                    if(std::filesystem::exists(labelfilename))  return false;
                                    return true; 
                                }),
                  list_images.end());

        std::cout << "Total found like " << img_ext << " " << this->list_images.size() << " files in folder: " << this->images_dir << std::endl;
    }
    else
    {
        LOG(ERROR) << "check directory return fail";
    }
}

torch::data::Example<> YOLODataset::get(size_t index)
{
    std::string sample_name = list_images[index];
    std::string image_name = std::filesystem::path(images_dir).append(sample_name + ".jpg").string();
    std::string label_name = std::filesystem::path(labels_dir).append(sample_name + ".txt").string();

    cv::Mat image = cv::imread(image_name, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    //std::cout << "YOLODataset index: " << index << " filename: " << list_images[index] << std::endl;

    std::vector<BBox_xyhw> bboxs;
    load_xyhw_labels(label_name, bboxs);

    float x_scale = float(width) / float(image.cols);
    float y_scale = float(height) / float(image.rows);

    auto norm_size = [&](int _width, int _height) {          // 后续修改为全局调用函数
        cv::resize(image, image, cv::Size(_width, _height)); // 图像规一到[width, height]
        for (int i = 0; i < bboxs.size(); i++)
        {
            // bboxs[i].scale(x_scale, y_scale);
            ///*
            //auto [x1, y1, x2, y2] = bboxs[i].getImageRect(_width, _height);
            //cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 250), 1, cv::LINE_8);
            //*/
        }
    };

    norm_size(width, height);

    //cv::imshow("show label", image);
    //cv::waitKey();

    torch::Tensor img_tensor = torch::from_blob(image.data, {height, width, 3}, torch::kByte)
                                   .permute({2, 0, 1}); // [h, w, c] ==>[c, h, w]
    int labels_num = bboxs.size();
    torch::Tensor label_tensor = torch::zeros({labels_num, 6}).to(torch::kFloat32);
    if(label_tensor.sizes().size()==1 || labels_num == 0)
    {
        LOG(WARNING) << "read boxs number 0, file: " << label_name;
    }

    // yolo需要的数据格式 [image_index = (0,暂时), cls_index, x, y, w, h ]
    for (int i = 0; i < labels_num; i++)
    {
        // 根据python代码测试，Yolov5-master中targets数据插入了个空数据。用于batch中指出是那张图
        label_tensor[i][2] = bboxs[i].x;
        label_tensor[i][3] = bboxs[i].y;
        label_tensor[i][4] = bboxs[i].w;
        label_tensor[i][5] = bboxs[i].h;
        label_tensor[i][1] = bboxs[i].cls_id;
    }

    return {img_tensor.clone(), label_tensor.clone()};
}

bool YOLODataset::check_dirs(const std::string &data_root, const std::string &image_dir)
{
    std::filesystem::path check_folder = std::filesystem::path(data_root).append(image_dir);
    std::cout << "check_folder: " << check_folder.string() << std::endl;
    if (std::filesystem::is_directory(check_folder))
    {
        this->images_dir = check_folder.string();
        std::string tmp = check_folder.filename().string(); // train2017
        this->labels_dir = check_folder.parent_path().parent_path().append("labels").append(tmp).string();

        std::cout << "Select dataset: " << tmp << std::endl;
        std::cout << "    images_dir: " << this->images_dir << std::endl;
        std::cout << "    labels_dir: " << this->labels_dir << std::endl;

        if (std::filesystem::exists(images_dir) && std::filesystem::exists(labels_dir))
            return true;
    }

    return false;
}
#include <fstream>
#include <sstream>
void YOLODataset::load_xyhw_labels(const std::string &filename, std::vector<BBox_xyhw> &bboxs)
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
