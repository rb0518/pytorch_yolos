#pragma once

#include <tuple>
#include <vector>
#include <variant>
#include <type_traits>
#include <utility>
#include <stdexcept>

template<typename Tuple, typename ArgVariant, size_t N>
struct vector_2_tuple
{
	static void apply(Tuple& t, const std::vector<ArgVariant>& args)
	{
		if ((N - 1) < args.size())
		{
			using TargetType = typename std::tuple_element<N - 1, Tuple>::type;
			std::get<N - 1>(t) = std::get<TargetType>(args[N - 1]);
		}
		vector_2_tuple<Tuple, ArgVariant, N - 1>::apply(t, args);
	}
};

template<typename Tuple, typename ArgVariant>
struct vector_2_tuple<Tuple, ArgVariant, 0>
{
	static void apply(Tuple& t, const std::vector<ArgVariant>& args) {}
};

/*
// Help vectorvaraintToTuple:
// 通过vector和variant实现多异种参数封装，达到统一接口封装，通过该函数实现具体数据的对应转换，简化代码
// 编程与减少出错
// 
// Achieve unified interface encapsulation by implementing multi-heterogeneous parameter 
// encapsulation through `vector` and `variant`, enabling the function to handle specific data 
// The corresponding conversion simplifies code programming and reduces errors
// 
// Usage: (see class ConvImpl in conv.h and conv.cpp )
//	
//  class ConvImpl : public BaseModule
//	{
//	public:
// 		using Conf_Params = std::tuple<int, int, int, int, int, int, std::string>;
//    	Conf_Params get_defaults() {
//     		return std::make_tuple(0, 1, 1, -1, 1, 1, std::string("True"));
//    	}	
//		...
//	
//	void ConvImpl::set_params(int in_channels, int number, std::vector<arg_complex>& args) 
//	{
//		Conf_Params defs = get_defaults();
//		vectorvariantToTuple<Conf_Params, arg_complex>(defs, args);
//		...
//	}
//		
*/
template<typename Tuple, typename ArgVariant>
void vectorvariantToTuple(Tuple& defaults, const std::vector<ArgVariant>& args)
{
	constexpr auto tuple_size = std::tuple_size<Tuple>::value;
	if (tuple_size < args.size())
		LOG(ERROR) << "vectorvariantToTuple: tuple size error: " << tuple_size << " args.size: " << args.size();
	
	vector_2_tuple<Tuple, ArgVariant, tuple_size>::apply(defaults, args);
	return;
}
