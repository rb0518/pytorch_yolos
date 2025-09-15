cmake_minimum_required(VERSION 3.12)

project(LibtorchCPP LANGUAGES C CXX)

# linux下可以通用命令行cmake -DCMAKE_BUILD_TYPE=Release快速编译为Release版本
# Win32下需要cmake --build . --config Release快速编译为Release版本
# 

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()


set(CMAKE_CXX_STANDARD 17)  # 设置C++标准为C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 确保要求该标准
set(CMAKE_CXX_EXTENSIONS OFF) 

if (MSVC)
    add_compile_options(/wd4100)  # 禁用未使用参数的警告 (C4100)
    add_compile_options(/wd4018)  # 禁用有符号/无符号比较警告 (C4018)
    add_compile_options(/wd4251)  
    add_compile_options(/wd4267)  
    add_compile_options(/wd4244)  
    add_compile_options(/wd4275)  
    add_compile_options(/wd4819)  
    add_compile_options(/wd4005)  
    add_compile_options(/wd4067)  
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)


# add opencv include and libs to project
if(WIN32)
    message("path to your own opencv folder. must can find OpenCVConfig.cmake")
    set(OpenCV_DIR "D:/thirdparty/opencv")
endif()  
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# add libtorch include and libs to project
if(WIN32)
# 修改cmake\Caffe2\public\cuda.cmake
#在第 59 行 find_package(CUDAToolkit REQUIRED) 之后添加两行内容：
#add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
#set_property(TARGET CUDA::nvToolsExt APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}")
# 还有错误，手动添加
    include_directories(
        D:/libtorch/include
        D:/libtorch/include/torch/csrc/api/include
    )
    file(GLOB TORCH_LIBRARIES "D:/libtorch/lib/*.lib")
elseif(UNIX)
    set(Torch_DIR "/home/michael/libtorch/share/cmake/Torch")
    find_package(Torch REQUIRED)
    message("${TORCH_INCLUDE_DIRS} ${TORCH_LIBRARIES}")
    include_directories(${TORCH_INCLUDE_DIRS})
endif()

if(WIN32)
    # windows下是手动编译放到了指定目录下，而linux下通过sudo apt install libgflags-dev libgoogle-glog-dev
    # 命令安装到了/usr/include /usr/lib/x86_x64_linux_gnu下
    set(GFLAGS_ROOT "D:/thirdparty/gflags")
    set(GFLAGS_INCLUDE_DIRS "${GFLAGS_ROOT}/include")
    set(GFLAGS_LIBRARIES ${GFLAGS_ROOT}/lib/gflags.lib)
    set(GLOG_ROOT "D:/thirdparty/glog")
    set(GLOG_INCLUDE_DIRS "${GLOG_ROOT}/include")
    set(GLOG_LIBRARIES ${GLOG_ROOT}/lib/glog.lib)
    include_directories(${GFLAGS_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS})
endif()

#------------------------------------------------------------------------------
# OpenGL
#------------------------------------------------------------------------------

include_directories(
    "${CMAKE_SOURCE_DIR}/include" 
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/models"
    "${CMAKE_SOURCE_DIR}/utils"
)

file(GLOB_RECURSE add_src_files LIST_DIRECTORIES true 
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.h" 
        "${CMAKE_SOURCE_DIR}/models/*.cpp"
        "${CMAKE_SOURCE_DIR}/models/*.h" 
        "${CMAKE_SOURCE_DIR}/utils/*.cpp"
        "${CMAKE_SOURCE_DIR}/utils/*.h"         
    )  

add_executable(yolos ${CMAKE_SOURCE_DIR}/main.cpp ${add_src_files})

if(WIN32)
    find_package(OpenGL REQUIRED)
    target_link_libraries(yolos opengl32)
endif()

# 解决libtorch在Windows下用VS2022编译问题，2.4+cuda12.6下没有问题，不需要进行处理
if (WIN32)
    target_compile_options(yolos PRIVATE /D_USE_MATH_DEFINES)  # 解决Windows数学常量问题
    target_link_options(yolos PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")  # 修复CUDA不可用问题:cite[1]:cite[2]
    if(MSVC)
        target_compile_definitions(yolos PRIVATE GLOG_USE_GLOG_EXPORT)
    endif()
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/external/)

if(WIN32)
    target_link_libraries(yolos ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES} yaml-cpp)
elseif(UNIX)
    target_link_libraries(yolos ${OpenCV_LIBS} ${TORCH_LIBRARIES}  gflags glog yaml-cpp)
endif()
