#pragma once

#include <vector>
#include <variant>
#include <charconv>

// args 根据is_num决定是用n or s 
// s: 要根据情况决定，包含了None True False nearest nc, anchors等
using arg_complex = std::variant<int, std::string, bool, float>;

// define for yaml settings
using VariantValue = std::variant<std::string, int, bool, float, std::vector<int>>;
using VariantConfigs = std::unordered_map<std::string, VariantValue>;

template<typename T>
T getVariantValue(VariantConfigs* pargs, std::string key, T defValue) {
    /*
        if (pargs->count(key) == 0)
            return defValue;

        auto& value = pargs->at(key);
        if constexpr (std::is_same_v<T, float>)
            return std::holds_alternative<float>(value) ? std::get<float>(value) : defValue;
        else if constexpr (std::is_same_v<T, bool>)
            return std::holds_alternative<bool>(value) ? std::get<bool>(value) : defValue;
        else if constexpr (std::is_same_v<T, int>)
            return std::holds_alternative<int>(value) ? std::get<int>(value) : defValue;
        else if constexpr (std::is_same_v<T, std::string>)
            return std::holds_alternative<std::string>(value) ? std::get<std::string>(value) : defValue;
        else
        {
            static_assert(always_false<T>::value_compare, "unsupport type");
        }
        */
    return pargs->count(key) > 0 ? std::get<T>(pargs->at(key)) : defValue;
}


inline std::tuple<float, bool> ConvertToNumber(const std::string& str)
{
    float f_r;
    auto result = std::from_chars(str.data(), str.data() + str.size(), f_r);
    return std::make_tuple(f_r, result.ec == std::errc());
}


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
            if (s.find('.') == std::string::npos)
                tmp = int(numb);
            else
                tmp = float(numb);
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
        if(args.size())
        {
            for(auto item : args) 
            {
                std::visit([&](auto&& arg){
                    ss_args << " " << arg;
                }, item);
            }
        }
        else
            ss_args << "[]";
        ss << std::setw(35) << ss_args.str();
        
        return ss.str();
    }
}YoloLayerDef;