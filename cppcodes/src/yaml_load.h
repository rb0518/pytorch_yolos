#pragma once

#include <yaml-cpp/yaml.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <tuple>

#include <variant>
#include <unordered_map>
#include <unordered_set>

#include "BaseModel.h"

// 判定指定名字是否在指定的
inline bool check_str_in_strs(const std::vector<std::string> names, const std::string& name){
    return std::count(names.begin(), 
                        names.end(), name);
}

std::tuple<float, bool> ConvertToNumber(const std::string& str);
// Yolo layer define [from number name args]

typedef struct tagYoloLayerDef
{
    std::vector<int>    froms;
    int                 number;
    std::string         name;
    std::vector<arg_complex> args;  //first record as strings

    void add_one_arg(const std::string s)
    {
        auto [numb,r_b] = ConvertToNumber(s);
        arg_complex tmp;
        if(r_b) // 是可以转换为数字
        {
            tmp = int(numb);
        }
        else
        {
            if(s == "False")    tmp = false;
            else if(s == "True") tmp = true;
            else tmp = s;
        }
        args.push_back(tmp);
    }

    std::string show_info()
    {
        std::stringstream ss;
        ss << std::setw(20) << std::left << name;
        std::stringstream ss_from;
        ss_from << "from: ";
        if(froms.size() == 1)
        {
            ss_from << std::setw(4) << froms[0];
        }
        else
        {
            for(auto item : froms)
                ss_from << std::setw(4) << item;
     
        }
        ss << std::setw(25)<< ss_from.str();

        ss << " number: " << std::setw(5) << number;
        
        std::stringstream ss_args;
        ss_args << "args: ";
        for(auto item : args) 
        {
            std::visit([&](auto&& arg){
                ss_args << " " << arg;
            }, item);
        }
        ss << std::setw(35) << ss_args.str();
        
        return ss.str();
    }
}YoloLayerDef;

void TestYamlRead(const std::string& yamlfile="./yolov5s.yaml", bool debug_show = true);
bool TestDataYamlRead(const std::string& yamlfile="../../data/coco128.yaml", bool debug_show = true);

using VariantValue = std::variant<std::string, int, bool, float, std::vector<int>>;\
/*
class OrderedConfig {
public:
    // 插入元素（保持插入顺序）
    void insert(const std::string& key, VariantValue value) {
        if (!map.count(key)) {
            order.push_back(key);
        }
        map[key] = std::move(value);
    }

    // 快速查找（O(1)复杂度）
    VariantValue* find(const std::string& key) {
        if (auto it = map.find(key); it != map.end())
            return &it->second;
        return nullptr;
    }

    // 顺序访问迭代器
    auto begin() const { return order.begin(); }
    auto end() const { return order.end(); }
    const VariantValue& at(const std::string& key) const { return map.at(key); }

private:
    std::vector<std::string> order;           // 保持键的插入顺序
    std::unordered_map<std::string, VariantValue> map;  // 提供快速查找
};
*/
 using VariantConfigs = std::unordered_map<std::string, VariantValue>;
// std::map auto sort by key, want keep the push order, use std::vector alter std::map
//using OptConfig = std::vector<std::pair<std::string, VariantValue>>;

VariantConfigs set_cfg_opt_default();
VariantConfigs load_cfg_yaml(const std::string& cfgs_file);
void save_cfg_yaml(const VariantConfigs& cfgs, const std::string& optfile);

VariantConfigs set_cfg_hyp_default();

void show_cfg_info(const std::string& title, const VariantConfigs& cfgs);

void read_data_yaml(const std::string& data_file, std::string& train_path, std::string& val_path, 
                std::vector<std::string>& names);
