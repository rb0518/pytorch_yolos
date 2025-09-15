#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "utils.h"

class YOLODataset : public torch::data::Dataset<YOLODataset>
{
public:
    explicit YOLODataset(const std::string& datacfg/*path to data/coco128.yaml*/,
        const std::string& img_ext/*".jpg"*/,
        int _width, int _height,
        bool _isTrain = true);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override{
        return list_images.size();
    }

    std::vector<ExampleType> get_batch(c10::ArrayRef<size_t> indices) override
    {
        std::vector<torch::Tensor> imgs;
        std::vector<torch::Tensor> labels;
//        std::cout << "indices: " << indices << std::endl;
        for (size_t idx : indices)
        {
//            std::cout << "idx : " << idx << std::endl;
            torch::data::Example<> sample = get(idx);
            auto img = sample.data;
            img = img.squeeze(0);
//            std::cout << "image " << img.sizes() << std::endl;
            imgs.push_back(img);

            auto label = sample.target;
//            std::cout << "label 0: " << label.sizes() << std::endl;
            if(label.sizes().size() == 2)
            {
                label.index_put_({"...", 0}, idx - indices[0]);
                //std::cout << "label " << label.sizes() << std::endl;
                labels.push_back(label);
            }
        }

        auto batch_imgs = torch::stack(imgs, 0);
        //std::cout << " batch imgs: " << batch_imgs.sizes() << std::endl;
        // Concatenate labels along existing dimension

        torch::Tensor batch_labels;
        if(labels.size() > 1)
            batch_labels = torch::cat(labels, 0);
        else if(labels.size() == 1)
            batch_labels = labels[0];
        else
            batch_labels = torch::zeros({1});
        //std::cout << " batch labels: " << batch_labels.sizes() << std::endl;

        std::vector<ExampleType> batch_ret;
        batch_ret.emplace_back(batch_imgs, batch_labels);
        return batch_ret;
    }

private:
    std::string                 images_dir;     //path/to/image
    std::string                 labels_dir;     //path/to/labels
    std::vector<std::string>    list_images;    // 只保存stem，要确保label与image同名
    int                         width;
    int                         height;
    bool isTrain = true;

    std::vector<std::string> class_names;
    bool check_dirs(const std::string& data_root, const std::string& image_dir);
    
    void load_xyhw_labels(const std::string& filename, std::vector<BBox_xyhw>& bbox);
};