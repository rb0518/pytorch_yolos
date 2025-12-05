#pragma once

#include <yaml-cpp/yaml.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <tuple>

#include <variant>
#include <unordered_map>
#include <unordered_set>

#include "globaldef.h"


// Yolo layer define [from number name args]
void TestYamlRead(const std::string& yamlfile="./yolov5s.yaml", bool debug_show = true);
bool TestDataYamlRead(const std::string& yamlfile="../../data/coco128.yaml", bool debug_show = true);
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
// std::map auto sort by key, want keep the push order, use std::vector alter std::map
//using OptConfig = std::vector<std::pair<std::string, VariantValue>>;

VariantConfigs set_cfg_opt_default();
VariantConfigs load_cfg_yaml(const std::string& cfgs_file);
void save_cfg_yaml(const VariantConfigs& cfgs, const std::string& optfile);

VariantConfigs set_cfg_hyp_default();

void show_cfg_info(const std::string& title, const VariantConfigs& cfgs);

void read_data_yaml(const std::string& data_file, std::string& train_path, std::string& val_path, 
                std::vector<std::string>& names);

inline bool check_str_in_strs(const std::vector<std::string> names, const std::string& name){
    return std::count(names.begin(), 
                        names.end(), name);
}                

inline std::string findSubstringInStrings(const std::string& mainStr,
    const std::vector<std::string>& substrings) {
    for (const auto& sub : substrings) {
        if (mainStr.find(sub) != std::string::npos) {
            return sub;
        }
    }
    return "";
}